# ShadowContact

Implementation of **Observable Multi-Frame Shadow Geometry for Reliability-Aware Object–Plane Clearance Estimation**.

ShadowContact trains a compact object/shadow perception model, jointly estimates clearance/light/pose from multiple calibrated frames, and reports information conditioning separately from image mismatch. A finite-budget boundary residual corrector is fitted before a separate frozen-estimator failure selector. Contact is a thresholded clearance event with validation-only probability calibration.

## Installation

```bash
python -m venv .venv
python -m pip install -e .
```

Use Python 3.11 or 3.12. Activate the virtual environment in your shell.

## Prepare calibrated clips

The end-to-end task needs metric camera, receiver-plane and object-template calibration. Export these from controlled Blender scenes or calibrated tabletop acquisition. Public SBU, ISTD and CVSD shadow datasets can train/evaluate perception, but their masks alone do not provide metric clearance targets.

Create `data/clips.csv` with `object_id,path,split,height`. `path` points to a local NPZ; `height` is measured clearance in millimetres. `split` is `train`, `select`, `cal` or `test`. Object instances must be disjoint. Keep materials, light/motion combinations and scene families grouped when creating compositional holdouts. Selection training (`select`) is separate from estimator training (`train`) and threshold/probability calibration (`cal`).

Each NPZ contains:

| Array | Meaning |
|---|---|
| `rgb` | T×H×W×3 uint8 frames |
| `template` | T×N×3 associated object boundary points at zero clearance, in receiver coordinates and millimetres |
| `K` | 3×3 calibrated intrinsics |
| `R`, `translation` | T×3×3 and T×3 world-to-camera transforms |
| `plane` | Four coefficients `[nx,ny,nz,d]` with unit normal and millimetre offset |
| `object_mask`, `shadow_mask` | T×H×W binary training annotations; optional on evaluation clips |

Sample the associated template contour uniformly before export and keep its point identity through the clip. The template is a calibrated geometric proxy, not ground-truth test clearance. A single associated object and dominant directional light are assumed per clip. Train masks come from annotation/rendering, and geometry inference uses only the learned RGB shadow predictions plus calibration. Post-hoc clearance truth never enters optimization.

## Train and estimate

```bash
shadowcontact data/clips.csv --output outputs/full
```

`--frames 8`, `--boundary-points 256`, `--initializations 4` and `--iterations 8` set frame use, symmetric contour sampling, coarse light starts and optimization budget. `--epochs 30` controls RGB segmentation training. `--residual-budget 4` bounds correction magnitude in pixels. `--contact-tolerance 2` fixes contact at clearance at most 2 mm. `--coverage 0.8` selects a validation risk cutoff. `--pose-sd` sets the pixel nuisance-shift prior; `--initial-height` and `--maximum-height` define initialization and the physically registered search range.

The plane-ray intersection projects the hypothesized 3-D contour into each image. Symmetric contour distances prevent small predicted contours from receiving a low one-sided loss. Huber least squares jointly fits height, two light coordinates and shared pixel pose offsets. The raw weighted image Jacobian determines the information spectrum and height uncertainty; pose regularization is excluded from that reported information matrix.

## Baselines, ablations and sweeps

```bash
shadowcontact data/clips.csv --variant single --output outputs/single
shadowcontact data/clips.csv --variant analytic --output outputs/analytic
shadowcontact data/clips.csv --variant no_selector --output outputs/no_selector
shadowcontact data/clips.csv --variant no_observability --output outputs/no_observability
shadowcontact data/clips.csv --residual-budget 0 --output outputs/no_residual
shadowcontact data/clips.csv --frames 4 --output outputs/four_frames
```

`single` uses one frame without residual correction. `analytic` keeps multi-frame geometry without the learned corrector. Other controls remove the selector, its observability inputs, or boundary correction. Use identical held-out object assignments and seeds. Vary motion/calibration/noise through independent scene exports; inference settings should be selected on calibration clips and then frozen.

## Results

`predictions.json` includes per-clip clearance, uncertainty, residual, minimum information eigenvalue, condition number, visible support, contact probability and acceptance. Rank-deficient height uncertainty is saved as `null`. `model.pkl` contains perception weights, residual model and calibration/selector parameters. `risk_coverage.csv` and `risk_coverage.png` show the selective contact-error curve. `perception.csv` reports shadow IoU, balanced error, boundary F1 within two pixels and backward-Farneback-flow-warped temporal shadow IoU where masks exist.

`metrics.json` reports contact macro-F1/balanced accuracy, all-clip and accepted-clip clearance MAE in millimetres, coverage, accepted contact risk, risk at 80% coverage, AURC and failure-detection AUROC. Failure is contact misclassification or clearance error above 10 mm. Each clip has equal weight. AURC is the mean cumulative risk over nonempty retained prefixes, avoiding an invented zero-risk point. The validation threshold can yield a different test coverage; tied calibrated risks are retained together. Empty acceptance and single-class AUROC populations are `null`.

## Implementation choices

Perception uses compact convolutional local/context features with BCE and Dice supervision. The residual is a ridge model of signed-distance and normalized contour-position features, bounded by a tanh and centered across the clip; it cannot read contact or test clearance. The geometry optimizer uses finite-difference trust-region least squares and reports the undamped image information spectrum. Selector inputs are residual, condition, height uncertainty, visibility and correction magnitude. A logistic failure predictor is trained on `select`, then isotonic probability/risk maps and the coverage cutoff are fitted on `cal`. A constant empirical selector is used only when selection-training outcomes contain one class. Generic pretrained depth/video systems are separate external baselines and are not renamed as geometry controls.
