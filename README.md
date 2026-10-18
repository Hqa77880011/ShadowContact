# ShadowContact

Implementation of **Observable Multi-Frame Shadow Geometry for Reliability-Aware Object–Plane Clearance Estimation**.

## Installation

Use Python 3.11 or 3.12. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

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

## Run and evaluate

```bash
shadowcontact data/clips.csv --output outputs/full --seed 0
```

## Ablations

Supported controls: `analytic`, `single`, `no_selector`, `no_observability`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
shadowcontact data/clips.csv --output outputs/analytic --seed 0 --variant analytic
```

## Outputs

`model.pkl` stores perception/correction/selector state. `predictions.json`, `perception.csv`, `risk_coverage.csv`, `metrics.json` and `risk_coverage.png` contain executed clearance/contact predictions and diagnostics.

Source inspection: Python 3.13.5, numpy 2.1.3, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, torch 2.7.1+cu118, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
