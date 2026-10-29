# CLAUDE.md — Monocular Free-Space Perception

Read this file fully before every task. It is the source of truth for how this
project is organised, what is already validated, and how Phase 5 is to be built.

## What this project is

One RGB camera in; a metric, top-down map of where a ground robot can safely drive
out. No LiDAR, no stereo. LiDAR (KITTI) is used only to train and to grade.

Pipeline for one image:

```
photo -> Depth Anything V2 (metric, fine-tuned on KITTI) -> depth map
      -> back-project with K -> RANSAC ground plane (n, d)
      -> metric scale s = camera_height / d          (refused if the plane fit is invalid)
      -> 3D points in a FLOOR frame (z = height above floor)
      -> height labels (ground / obstacle / hole / ignore)
      -> optional semantic fusion (SegFormer-B0 Cityscapes): flat counts as free
         only if the pixel's class is drivable
      -> 3-state BEV grid (FREE / OCCUPIED / UNKNOWN) -> inflate by robot radius
      -> A* path
```

The owner (Bhaavin) is a mechanical engineering student building this as a
portfolio project for CV/perception internships. He is learning as he builds:
explain WHY, in plain words, for every non-trivial change.

## Status: Phases 1-4 are done and measured

| Phase | Result (KITTI drive 2011_09_26_0023, val) |
|---|---|
| 1 Geometry | LiDAR projection matches KITTI depth GT: median 2.1 cm |
| 2 Metric depth | Fine-tuned AbsRel 0.071 (pretrained 0.101). Camera-height scale within 0.3% of oracle (median s/s_oracle 1.003) |
| 3 BEV grid | Free-space IoU 0.830, false-free 7.6% (1.8% after inflation) |
| 4 Semantics + A* | road+sidewalk fusion: IoU 0.816, false-free 4.4%, 0.3% after inflation, missed obstacles 16.4% -> 8.8% |

Results live in `report/` (`*.json`, `*.txt`) and `report/image/`.

## Repository map

Validated core — DO NOT MODIFY without asking first. Phase 5 must IMPORT these,
not rewrite them:

| File | Role |
|---|---|
| `camera.py` | calibration parsing, transforms, project / backproject, LiDAR -> depth map |
| `kitti_loader.py` | KITTI raw drive loading (KITTI only; not used by the server) |
| `ground_plane.py` | RANSAC plane fit with noise-adaptive refinement, validity check |
| `scale.py` | `estimate_metric_scale`: depth -> plane -> s = h / d |
| `metrics.py` | depth metrics |
| `pointcloud.py` | `floor_frame(n, d)`, `classify_heights` |
| `bev.py` | `build_grid`, `cell_index`, `visible_mask`, `inflate`, `lidar_to_bev` |
| `planner.py` | `astar`, `pick_start_goal` |

PyTorch-based (training / evaluation only — the server must NOT import these):
`depth_model.py`, `segmenter.py`, `finetune_depth.py`.

Evaluation scripts (KITTI): `validate_projection.py`, `explore_depth.py`,
`check_ground_plane.py`, `eval_depth.py`, `eval_freespace.py`.
`eval_freespace.py` contains the reference implementation of the full pipeline,
including `upsample_rows` (4x row upsampling before back-projection — required,
otherwise far rows leave gaps of UNKNOWN cells) and `paint_photo`.

## Data and environment (the owner's machine)

- Windows, VS Code, `cmd` terminal. Virtual env `.venv` (Python 3.12). Activate with
  `.venv\Scripts\activate`. Run everything from the project root.
- GPU: RTX 4060 Laptop, 8 GB. PyTorch with CUDA is installed (training/export only).
- Data (never commit):
  - `data/2011_09_26/caliberation files/` — calibration (note the spelling; it is
    intentional, `kitti_loader.CALIB_FOLDER` matches it)
  - `data/2011_09_26/2011_09_26_drive_0023_sync/` — val drive, full
  - other `2011_09_26_drive_XXXX_sync/` — train drives, `image_02` only
  - `data/kitti_depth/{train,val}/` — KITTI depth benchmark ground truth
- Weights (never commit): `checkpoints/depth_best.pt` (fine-tuned depth, state_dict
  for `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`).
- `.gitignore` already excludes `data/`, `checkpoints/`, `*.pt`, `*.onnx`, `.venv/`.

## Coding preferences — follow these

- Minimal code. No helper functions or classes when a simple built-in approach works.
  No speculative abstractions, no unused options.
- Scripts use argparse with sensible defaults (exception: `finetune_depth.py` uses
  constants at the top).
- Results are saved under `report/` (text or JSON) and `report/image/` (pictures).
- Comments explain WHY, not what. Docstrings at the top of each file explain the idea.
- Test the smallest thing first (one image, one endpoint) before scaling up.

## Hard rules

1. Never commit data, weights, ONNX files, or secrets.
2. `git commit` before starting each milestone, so every milestone is one reviewable diff.
3. Ask before modifying any validated core file, adding a dependency, or changing a
   metric definition.
4. Never hide a refused plane fit. If the fit is invalid, return no metric output and
   show the reason.
5. Do not claim performance you did not measure. Every speed or accuracy number in the
   UI or README must come from a script in this repo.
6. After each milestone, summarise in plain words: what was built, how it was checked,
   what the numbers were, what is still unverified.

## Phase 5 — goal

A public URL where anyone uploads a photo, sets the camera height, and gets back the
depth map, the occupancy grid and a planned path, with the assumptions stated on screen.

## Phase 5 — decisions already made

- Backend: FastAPI. Frontend: React + Vite, deployed on Vercel.
- Inference: ONNX Runtime on CPU. No PyTorch in the server or its Docker image.
- Backend hosting: Hugging Face Spaces, Docker SDK (app listens on port 7860).
- ONNX model files live in a Hugging Face Hub model repo; the Space downloads them at
  start-up. They are not in git.
- Intrinsics for uploaded photos (they have no calibration):
  1. apply the EXIF orientation first (Pillow `ImageOps.exif_transpose`);
  2. if EXIF has `FocalLengthIn35mmFilm` (f35): `fx = f35 / 36 * image_width`;
  3. otherwise use a horizontal FOV, default 70 deg, editable by the user:
     `fx = (W / 2) / tan(hfov / 2)`;
  4. `fy = fx`, `cx = W / 2`, `cy = H / 2`. Report which source was used.
- Inputs: image (reject > 10 MB; downscale the long side to a size chosen from the
  benchmark), `camera_height` (m, default 1.5), optional `hfov_deg`,
  `drivable` in {none, road, road+sidewalk}, `preset` in {outdoor, indoor}.
- Grid presets: outdoor = 20 m ahead x 20 m wide, 10 cm cells (KITTI setting);
  indoor = 10 m x 10 m, 5 cm cells.
- Any resize, crop or pad of the image must update K accordingly.
- Model licences: check the Hugging Face model cards and state them in the README.

## Phase 5 — milestones (do them in order; each must pass before the next)

**M1. `pipeline.py` — one function, no KITTI, no LiDAR, models passed in.**
`run(image, K, camera_height, depth_fn, seg_fn=None, drivable=..., preset=...)` ->
dict with: refusal reason or scale s, plane tilt, depth map, grid, inflated grid,
path (metres), painted photo, timings. Reuse the validated core and the logic in
`eval_freespace.py`; do not duplicate it more than necessary.
Check: on KITTI drive 0023 frame 10 (cropped as in `eval_depth.crop_bottom_centre`,
KITTI K, h = 1.65, PyTorch models), the grid equals the one `eval_freespace.py`
builds for the same frame.

**M2. `export_onnx.py` — both models to ONNX, with a parity check.**
Depth (fine-tuned weights) and SegFormer-B0. Prefer one fixed input size if dynamic
shapes do not export cleanly (DINOv2 interpolates position embeddings). Save the
normalisation mean/std next to each model (JSON).
Check on 20 KITTI val frames: depth median relative difference < 0.5% and max < 2%
versus PyTorch; segmentation label agreement > 99.5%. Save to `report/onnx_parity.json`.

**M3. `benchmark.py` — CPU latency.**
Time each stage (depth, segmentation, plane + grid + A*) on CPU with ONNX Runtime at
2-3 input sizes. Choose the shipping size from the numbers. Save to
`report/benchmark.json`. The chosen latency is what the UI displays.

**M4. FastAPI backend (`backend/`).**
`GET /health`; `POST /predict` (multipart) returning JSON: status, refusal reason,
scale, tilt, FOV used and its source, timings, depth and painted photo as PNG
(base64), the grid as base64 uint8 + shape + cell size, and the path in metres
(the frontend draws the grid on a canvas). Input validation and size caps. CORS for
localhost and the Vercel domain.
Check: KITTI frame with h = 1.65 and hfov = 80 deg gives a sensible grid; a photo of
a wall returns a refusal with its reason; oversized and non-image uploads are rejected.

**M5. Docker + Hugging Face Space.**
`python:3.12-slim`, onnxruntime, numpy, opencv-python-headless, pillow, fastapi,
uvicorn, huggingface_hub. Run locally first, then deploy.
Check: public `/health` responds; `/predict` works on the KITTI test image; image size
is reported.

**M6. React + Vite frontend (`web/`), on Vercel.**
Upload; camera height (default 1.5 m, with a note that it sets the scale); FOV
(pre-filled from EXIF when available, editable); drivable mode; preset. Panels:
photo with the grid painted on it, depth map, BEV grid on a canvas with the path,
camera marker and distance rings. Show: measured processing time, refusal reason
when the plane fit fails, a "waking up" state for Space cold starts, and the
assumptions (camera height sets scale, FOV source, flat floor, trained on German
streets). API URL from `VITE_API_URL`.
Check: works locally against the local API, then on Vercel against the Space.

**M7. Polish (optional, only after M6 works end to end).**
Mobile layout, webcam capture, `/predict/sequence`.

## Useful commands (existing evaluation)

```
python validate_projection.py --frame 10 --depth_root data\kitti_depth
python check_ground_plane.py --frames 10 50 100
python eval_depth.py --depth_root data\kitti_depth --tag finetuned --weights checkpoints\depth_best.pt
python eval_freespace.py --depth_root data\kitti_depth --weights checkpoints\depth_best.pt --drivable road+sidewalk --tag finetuned_road_sidewalk
```