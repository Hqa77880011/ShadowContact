import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.special import ndtr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


class Perception(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Conv2d(3, 24, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(24, 32, 3, padding=1), torch.nn.ReLU())
        self.context = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 32, 3, padding=1), torch.nn.ReLU())
        self.head = torch.nn.Conv2d(64, 2, 1)

    def forward(self, images):
        local = self.encoder(images)
        context = torch.nn.functional.interpolate(self.context(local), size=local.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(torch.cat([local, context], 1))


def project(theta, clip):
    h, lx, ly, dx, dy = theta
    normal = clip["plane"][:3]
    normal = normal / np.linalg.norm(normal)
    light = np.array([lx, ly, 1.])
    denominator = float(normal @ light)
    if abs(denominator) < 1e-4:
        denominator = np.copysign(1e-4, denominator or 1.)
    points = clip["template"] + h * normal
    distances = (points @ normal + clip["plane"][3]) / denominator
    shadow = points - distances[..., None] * light
    camera = np.einsum("tij,tnj->tni", clip["R"], shadow) + clip["translation"][:, None]
    homogeneous = np.einsum("ij,tnj->tni", clip["K"], camera)
    pixels = homogeneous[..., :2] / homogeneous[..., 2:].clip(1e-5)
    return pixels + np.array([dx, dy])


def boundary(mask):
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    edge = mask & (inside <= 1)
    return outside - inside, np.argwhere(edge)[:, ::-1]


def infer_geometry(clip, masks, args, corrector=None):
    count = min(args.frames, len(masks))
    clip = {k: v[:count] if k in ("template", "R", "translation") else v for k, v in clip.items()}
    sdf, contours = zip(*(boundary(m > .5) for m in masks[:count]))
    if any(len(c) < 3 for c in contours):
        return dict(height=0., sigma=float("inf"), residual=1e6, minimum_information=0., condition=1e12, support=0., correction=0., theta=[0.] * 5)
    sdf = np.stack(sdf)
    def correction_features(pixels):
        features = []
        for t, uv in enumerate(pixels):
            distance = map_coordinates(sdf[t], [uv[:, 1], uv[:, 0]], order=1, mode="nearest")
            features.append(np.c_[np.tanh(distance / 4), uv[:, 0] / sdf.shape[2] - .5, uv[:, 1] / sdf.shape[1] - .5, np.ones(len(uv))])
        return np.stack(features)
    def image_residual(theta, return_details=False):
        pixels = project(theta, clip)
        feature = correction_features(pixels)
        delta = np.zeros_like(pixels)
        if corrector is not None and args.variant not in ("analytic", "single"):
            delta = args.residual_budget * np.tanh(corrector.predict(feature.reshape(-1, feature.shape[-1])).reshape(pixels.shape))
            delta -= delta.mean((0, 1), keepdims=True)
            maximum_norm = np.linalg.norm(delta, axis=-1).max()
            delta *= min(1, args.residual_budget / max(maximum_norm, 1e-8))
        pixels = pixels + delta
        residual = []
        support = []
        for t, uv in enumerate(pixels):
            valid = (uv[:, 0] >= 0) & (uv[:, 0] < sdf.shape[2]) & (uv[:, 1] >= 0) & (uv[:, 1] < sdf.shape[1])
            support.extend(valid)
            forward = map_coordinates(sdf[t], [uv[:, 1], uv[:, 0]], order=1, mode="nearest")
            distance, _ = cKDTree(uv).query(contours[t][::max(1, len(contours[t]) // args.boundary_points)])
            distance = np.interp(np.linspace(0, 1, len(uv)), np.linspace(0, 1, len(distance)), distance)
            residual.extend(np.r_[forward, distance])
        if return_details:
            return np.asarray(residual), float(np.mean(support)), delta, feature, pixels
        return np.asarray(residual)
    def objective(theta):
        return np.r_[image_residual(theta), theta[3:] / args.pose_sd]
    best = None
    for angle in np.linspace(0, 2 * np.pi, args.initializations, endpoint=False):
        initial = np.array([args.initial_height, .7 * np.cos(angle), .7 * np.sin(angle), 0, 0])
        result = least_squares(objective, initial, bounds=([0, -5, -5, -5, -5], [args.maximum_height, 5, 5, 5, 5]), loss="huber", f_scale=2, max_nfev=args.iterations * 10)
        if best is None or result.cost < best.cost:
            best = result
    residual, support, delta, feature, pixels = image_residual(best.x, True)
    jac = best.jac[:len(residual)]
    weights = np.minimum(1, 2 / np.maximum(np.abs(residual), 1e-8))
    information = jac.T @ (weights[:, None] * jac)
    eigenvalues = np.linalg.eigvalsh(information)
    noise = np.mean(residual ** 2)
    covariance = np.linalg.pinv(information, rcond=1e-10) * noise
    rank = np.linalg.matrix_rank(information, tol=1e-8)
    sigma = np.sqrt(max(0, covariance[0, 0])) if rank == len(best.x) else float("inf")
    return dict(height=float(best.x[0]), sigma=float(sigma), residual=float(np.mean(np.abs(residual))), minimum_information=float(max(0, eigenvalues[0])), condition=float(eigenvalues[-1] / max(eigenvalues[0], 1e-12)), support=support, correction=float(np.mean(np.linalg.norm(delta, axis=-1))), theta=best.x.tolist())


def load_clip(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def frontend_masks(model, clip):
    with torch.no_grad():
        rgb = torch.tensor(clip["rgb"].transpose(0, 3, 1, 2), dtype=torch.float32) / 255
        return torch.sigmoid(model(rgb)).numpy()


def selector_features(results):
    return np.array([[np.log1p(r["residual"]), np.log1p(r["condition"]), np.log1p(min(r["sigma"], 1e3)), 1 - r["support"], r["correction"]] for r in results])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--output", default="outputs/full")
    p.add_argument("--variant", choices=["full", "analytic", "single", "no_selector", "no_observability"], default="full")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--boundary-points", type=int, default=256)
    p.add_argument("--initializations", type=int, default=4)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--residual-budget", type=float, default=4)
    p.add_argument("--pose-sd", type=float, default=1)
    p.add_argument("--initial-height", type=float, default=10)
    p.add_argument("--maximum-height", type=float, default=100)
    p.add_argument("--contact-tolerance", type=float, default=2)
    p.add_argument("--coverage", type=float, default=.8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.variant == "single":
        args.frames = 1
    manifest = pd.read_csv(args.manifest)
    if manifest.groupby("object_id").split.nunique().max() != 1:
        raise ValueError("Objects must not occur across splits")
    if not all((manifest.split == s).any() for s in ("train", "select", "cal", "test")):
        raise ValueError("Supply object-disjoint train, select, cal and test clips")
    torch.manual_seed(args.seed)
    model = Perception()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    for _ in range(args.epochs):
        for row in manifest[manifest.split == "train"].itertuples():
            clip = load_clip(row.path)
            images = torch.tensor(clip["rgb"].transpose(0, 3, 1, 2), dtype=torch.float32) / 255
            target = torch.tensor(np.stack([clip["object_mask"], clip["shadow_mask"]], axis=1), dtype=torch.float32)
            for start in range(0, len(images), 2):
                logits = model(images[start:start + 2])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target[start:start + 2])
                probabilities = logits.sigmoid()
                loss += 1 - (2 * (probabilities * target[start:start + 2]).sum() + 1) / (probabilities.sum() + target[start:start + 2].sum() + 1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    model.eval()
    correction_x, correction_y = [], []
    for row in manifest[manifest.split == "train"].itertuples():
        clip = load_clip(row.path)
        masks = frontend_masks(model, clip)[:, 1]
        geometry = infer_geometry(clip, masks, args)
        if geometry["support"] == 0:
            continue
        count = min(args.frames, len(masks))
        reduced = {k: v[:count] if k in ("template", "R", "translation") else v for k, v in clip.items()}
        theta = np.array(geometry["theta"])
        pixels = project(theta, reduced)
        for t, uv in enumerate(pixels):
            sdf, contour = boundary(masks[t] > .5)
            _, target_contour = boundary(clip["shadow_mask"][t].astype(bool))
            if len(target_contour) < 3:
                continue
            distances = map_coordinates(sdf, [uv[:, 1], uv[:, 0]], mode="nearest")
            _, ids = cKDTree(target_contour).query(uv)
            difference = target_contour[ids] - uv
            difference = np.arctanh(np.clip(difference / max(args.residual_budget, 1e-6), -.95, .95))
            correction_x.extend(np.c_[np.tanh(distances / 4), uv[:, 0] / sdf.shape[1] - .5, uv[:, 1] / sdf.shape[0] - .5, np.ones(len(uv))])
            correction_y.extend(difference)
    corrector = Ridge(alpha=1.).fit(correction_x, correction_y) if correction_x and args.residual_budget > 0 else None
    rows, masks_metrics = [], []
    for row in manifest[manifest.split != "train"].itertuples():
        clip = load_clip(row.path)
        masks = frontend_masks(model, clip)
        result = infer_geometry(clip, masks[:, 1], args, corrector)
        result.update(case=str(row.path), split=row.split, truth=float(row.height), true_contact=int(row.height <= args.contact_tolerance))
        variance = max(result["sigma"], .01)
        result["raw_probability"] = float(ndtr((args.contact_tolerance - result["height"]) / variance))
        rows.append(result)
        if "shadow_mask" in clip:
            truth = clip["shadow_mask"].astype(bool)
            prediction = masks[:, 1] > .5
            intersection = np.sum(truth & prediction)
            union = np.sum(truth | prediction)
            positive = truth
            negative = ~truth
            ber = .5 * ((~prediction[positive]).mean() if positive.any() else 0) + .5 * (prediction[negative].mean() if negative.any() else 0)
            boundary_scores = []
            for a, b in zip(prediction, truth):
                _, pa = boundary(a)
                _, pb = boundary(b)
                if not len(pa) or not len(pb):
                    boundary_scores.append(0.)
                    continue
                precision = (cKDTree(pb).query(pa)[0] <= 2).mean()
                recall = (cKDTree(pa).query(pb)[0] <= 2).mean()
                boundary_scores.append(2 * precision * recall / max(precision + recall, 1e-8))
            temporal = []
            for t in range(1, len(prediction)):
                current = cv2.cvtColor(clip["rgb"][t], cv2.COLOR_RGB2GRAY)
                previous = cv2.cvtColor(clip["rgb"][t - 1], cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(current, previous, None, .5, 3, 15, 3, 5, 1.2, 0)
                yy, xx = np.mgrid[:current.shape[0], :current.shape[1]].astype(np.float32)
                warped = cv2.remap(prediction[t - 1].astype(np.float32), xx + flow[..., 0], yy + flow[..., 1], cv2.INTER_LINEAR) > .5
                temporal.append(np.sum(warped & prediction[t]) / max(1, np.sum(warped | prediction[t])))
            masks_metrics.append(dict(case=str(row.path), split=row.split, iou=float(intersection / max(1, union)), ber=float(ber), boundary_f1=float(np.mean(boundary_scores)), flow_warped_temporal_iou=float(np.mean(temporal)) if temporal else None))
    select_rows = [r for r in rows if r["split"] == "select"]
    calibration_rows = [r for r in rows if r["split"] == "cal"]
    test_rows = [r for r in rows if r["split"] == "test"]
    probability_cal = IsotonicRegression(out_of_bounds="clip").fit([r["raw_probability"] for r in calibration_rows], [r["true_contact"] for r in calibration_rows])
    for r in rows:
        r["probability"] = float(probability_cal.predict([r["raw_probability"]])[0])
        r["error"] = int((r["probability"] >= .5) != r["true_contact"])
        r["failure"] = int(r["error"] or abs(r["truth"] - r["height"]) > 10)
    features = selector_features(select_rows)
    if args.variant == "no_observability":
        features[:, 1:3] = 0
    target = np.array([r["failure"] for r in select_rows])
    selector = LogisticRegression(max_iter=1000).fit(features, target) if len(np.unique(target)) == 2 else None
    def score(records):
        f = selector_features(records)
        if args.variant == "no_observability":
            f[:, 1:3] = 0
        return selector.predict_proba(f)[:, 1] if selector else np.full(len(records), target.mean())
    cal_score = score(calibration_rows)
    risk_map = IsotonicRegression(out_of_bounds="clip").fit(cal_score, [r["failure"] for r in calibration_rows])
    threshold = float(np.quantile(risk_map.predict(cal_score), args.coverage))
    test_score = risk_map.predict(score(test_rows))
    for r, s in zip(test_rows, test_score):
        r["rejection_score"] = float(s)
        r["accepted"] = bool((s <= threshold and np.isfinite(r["sigma"]) and r["support"] > .2) or args.variant == "no_selector")
    errors = np.array([r["error"] for r in test_rows])
    clearance_error = np.array([abs(r["height"] - r["truth"]) for r in test_rows])
    order = np.argsort(test_score)
    coverage = np.arange(1, len(order) + 1) / len(order)
    curve = np.cumsum(errors[order]) / np.arange(1, len(order) + 1)
    accepted = np.array([r["accepted"] for r in test_rows])
    metrics = dict(macro_f1=float(f1_score([r["true_contact"] for r in test_rows], [r["probability"] >= .5 for r in test_rows], average="macro", zero_division=0)), balanced_accuracy=float(balanced_accuracy_score([r["true_contact"] for r in test_rows], [r["probability"] >= .5 for r in test_rows])), clearance_mae_mm=float(clearance_error.mean()), coverage=float(accepted.mean()), accepted_clearance_mae_mm=float(clearance_error[accepted].mean()) if accepted.any() else None, accepted_contact_risk=float(errors[accepted].mean()) if accepted.any() else None, risk_at_80_coverage=float(curve[max(0, int(np.ceil(.8 * len(order))) - 1)]), aurc=float(np.mean(curve)), failure_detection_auroc=float(roc_auc_score([r["failure"] for r in test_rows], test_score)) if len(set(r["failure"] for r in test_rows)) == 2 else None)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if not np.isfinite(r["sigma"]):
            r["sigma"] = None
    (out / "predictions.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    pd.DataFrame(masks_metrics).to_csv(out / "perception.csv", index=False)
    pd.DataFrame(dict(coverage=coverage, risk=curve)).to_csv(out / "risk_coverage.csv", index=False)
    with (out / "model.pkl").open("wb") as f:
        pickle.dump(dict(perception=model.state_dict(), corrector=corrector, probability_calibrator=probability_cal, selector=selector, risk_calibrator=risk_map, threshold=threshold, arguments=vars(args)), f)
    plt.plot(coverage, curve, color="#4d8796")
    plt.xlabel("Coverage")
    plt.ylabel("Contact error risk")
    plt.tight_layout()
    plt.savefig(out / "risk_coverage.png", dpi=180)


if __name__ == "__main__":
    main()
