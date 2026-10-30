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

Phase 5 (torch-free unless noted):

| File | Role |
|---|---|
| `pipeline.py` | `run()`: the whole pipeline for one image, models passed in as functions; also `upsample_rows`, `fuse_labels`, `paint_photo`, `onnx_fn`, `PRESETS` |
| `check_pipeline.py` | M1 check (imports torch): `run()` grid == eval_freespace grid |
| `export_onnx.py` | M2 (imports torch): ONNX export + parity; patches the Depth Anything head for dynamic shapes |
| `benchmark.py` | M3: CPU latency + KITTI accuracy per input size |
| `backend/app.py` | M4: FastAPI server (`uvicorn backend.app:app --port 7860` from the root) |
| `backend/make_samples.py` | M4: builds the KITTI gallery (`python -m backend.make_samples`) |
| `backend/check_api.py` | M4 check against a running server (`python -m backend.check_api`) |

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
7. Commit messages: no `Co-Authored-By` trailers and no mention of Claude.

## Phase 5 — goal

A public URL where anyone uploads a photo, sets the camera height, and gets back the
depth map, the occupancy grid and a planned path, with the assumptions stated on screen.

## Phase 5 — decisions already made

- Backend: FastAPI. Frontend: React + Vite, deployed on Vercel.
- Inference: ONNX Runtime on CPU. No PyTorch in the server or its Docker image.
- Backend hosting: **Azure Container Apps** (Azure for Students), scale-to-zero, image
  pulled from **ghcr.io**; app listens on port 7860. (Was: Hugging Face Docker Space —
  blocked in M5: HF now requires a PRO subscription for Docker Spaces, see M5 status.)
- ONNX model files and the gallery live in the PUBLIC Hugging Face Hub model repo
  `BhaavinJain/monocular-freespace-models`; the Docker image downloads them at BUILD
  time, pinned to one repo commit (`MODEL_REVISION` in `space/Dockerfile`), so a cold
  start never downloads models. They are not in git.
- Intrinsics for uploaded photos (they have no calibration):
  1. apply the EXIF orientation first (Pillow `ImageOps.exif_transpose`);
  2. a FOV the user sends (`hfov_deg`) wins: `fx = (W / 2) / tan(hfov / 2)`. The UI
     pre-fills that field from EXIF, so an edited value must override EXIF;
  3. else, if EXIF has `FocalLengthIn35mmFilm` (f35):
     `fx = f35 * sqrt(W^2 + H^2) / 43.27` — the 35 mm-equivalent focal length is
     defined by the frame DIAGONAL (43.27 mm), so this holds for any aspect ratio
     and orientation;
  4. else the default horizontal FOV of 70 deg, with the formula of step 2;
  5. `fy = fx`, `cx = W / 2`, `cy = H / 2`. Report which source was used
     (`user` / `exif_35mm` / `default_70`; samples: `kitti_calibration`).
- Inputs: image (reject > 10 MB; resize to the size chosen from the benchmark —
  294 x 392, provisional, see "decisions after M3"), `camera_height` (m, default 1.5), optional `hfov_deg`,
  `drivable` in {none, road, road+sidewalk}, `preset` in {outdoor, indoor}.
- Grid presets: outdoor = 20 m ahead x 20 m wide, 10 cm cells (KITTI setting);
  indoor = 10 m x 10 m, 5 cm cells.
- Any resize, crop or pad of the image must update K accordingly.
- Model licences: check the Hugging Face model cards and state them in the README.

## Phase 5 — decisions after M3 (these override the M4 and M6 text where they differ)

**Shipping size: 294 x 392 (H x W, 4:3), provisional, chosen on KITTI only.**
Measured by `benchmark.py` (`report/benchmark.json`): the most accurate candidate
within the 2 s budget on 2 threads. Its depth AbsRel of 0.056 is a KITTI-only
number (10 frames of drive 0023, KITTI camera, h = 1.65 m) and must NEVER be
presented as the app's accuracy on other cameras — not in the UI, not in the README.
Laptop timings are not stable: the same configuration measured 0.59 s and 1.48 s
in two runs as the CPU clock changed, so the latency shown in the UI must come from
the M5 measurement on the deployed service (Azure Container Apps).

**Cross-camera accuracy is unmeasured.** Every KITTI test had fx between 565 and
995 px; a 70 deg photo at 392 px wide has fx ~280 px, outside anything tested.
On KITTI most of the error at other sizes is the camera-height scale, not the
depth shape (s/s_oracle 0.86-0.98 by size), so the scale is what to watch.
Planned for **Phase 6**: evaluate the pipeline on NYU Depth v2 and/or DIODE
(different cameras, ground-truth depth, known intrinsics). Camera height for each
image is derived by fitting the existing RANSAC plane (`ground_plane.fit_ground`)
to the GROUND-TRUTH floor depth. Report depth AbsRel and free-space metrics, and
revisit the input size with those numbers — including resizing so that fx falls
inside the tested range. No phone validation set.

**App structure:**
- Default view: a gallery of a few KITTI val frames bundled with the app, with
  their real K and h = 1.65 m. These results are backed by measurements.
- Upload mode: clearly labelled "experimental", stating that camera height and FOV
  are user-supplied and that accuracy on other cameras is not yet measured (after
  Phase 6: "measured on NYU/DIODE: ...", with the numbers from that script).
- M4 backend: add `GET /samples` (lists the gallery frames) and let `/predict`
  accept either a sample id or an uploaded image.
- The README must credit KITTI and state its licence for the bundled frames.

## Phase 5 — M4 definitions (implemented in `pipeline.describe` / `backend/app.py`; M6 must follow)

**Input sizing.** Every image goes through `pipeline.fit_to_size`: the largest crop
of the target aspect ratio, ANCHORED AT THE BOTTOM and centred sideways (keeps
the floor near the robot, drops sky or sides), then resized; K follows both.
Landscape -> 294 x 392, portrait -> 392 x 294 (same pixel count, never rotated).
Uploads must be at least 294 px on the short side (no upscaling).

**Verdict** (computed on the raw grid, not the inflated one):
- *corridor*: grid columns with |y| <= robot radius (the strip the robot's body sweeps).
- `visible_from_m`: first grid row where every corridor cell is in the camera's
  view (`bev.visible_mask`). The blind zone before it (~6 m on KITTI) is never seen.
- `corridor_clear_m`: from that row forward, where the first non-FREE corridor
  cell starts (x_max if none).
- `nearest_obstacle`: nearest OCCUPIED cell (by forward row, then nearest the
  centre line) within the corridor widened by 1 m each side (`SIDE_MARGIN`):
  `distance_m` (near edge of the cell), `lateral_m`, `side` (ahead if inside the
  corridor, else left/right), `class` = majority Cityscapes class of the points
  that made the cell OCCUPIED when segmentation is on, else "obstacle". It can be
  closer than `visible_from_m`: the floor there is unseen, but an object's upper
  part can be.
- `level` and `text`:
  - `blocked` when the corridor's free run ends at an OCCUPIED cell: "Stop: <class> <d> m ahead"
    (the blocking cell nearest the centre line);
  - `clear`: "Clear from <a> m to <b> m ahead; the floor in the first <a> m is outside
    the camera's view." plus, if there is one, "Nearest obstacle: <class> <d> m ahead,
    <y> m to the <side>." ("the floor": an object's upper part can be seen closer
    than <a>, so "nearest obstacle 5.4 m" next to "first 6 m" is not a contradiction);
  - `unknown`: the corridor is never fully visible, or nothing in it is known to be free;
  - `refused`: "No floor found: <reason>" (no metric fields at all).
- Also `path_reach_m` (farthest forward point of the A* path) and `path_length_m`.

**Path ribbon** (`ribbon_px`): the A* path as a band 2 x robot radius wide: each
path point offset by the radius perpendicular to the direction of travel, on the
floor (z = 0), projected with K and the floor transform. One polygon (left edge
out, right edge back) in model-image pixels. The path is NOT burned into
`painted_png`; the frontend draws the ribbon.

**Distance lines**: straight floor lines at fixed FORWARD distances (outdoor
5/10/15/20 m, indoor 2.5/5/7.5/10 m; `PRESETS[...]["marks"]`). Photo: projected
polylines (`distance_lines_px`, only the in-image part; lines inside the blind
zone are empty). BEV map: horizontal lines at the same distances
(`distance_marks_m`), drawn by the frontend. No rings.

**Hover data** (`hover`): per pixel of the model image, `depth_cm` (uint16),
`labels` (uint8: 0 ground, 1 obstacle, 2 hole, 3 ignored, after fusion) and
`classes` (uint8 Cityscapes id, or null without segmentation), all base64
row-major, plus `K` and `T_floor_cam`. The frontend back-projects the hovered
pixel (`X = (u - cx) Z / fx`, `Y = (v - cy) Z / fy`), applies `T_floor_cam` to get
forward / left / height, and looks up that point's BEV cell (and the reverse for
hovering the map).

**Gallery** (`backend/make_samples.py` -> `backend/samples/`, git-ignored, to the
Hub in M5): 6 frames of drive 0023 taken from `benchmark.py`'s frame list. Each:
the 375 x 500 bottom-centre 4:3 crop (fit_to_size turns it into exactly the
input benchmark.py measured), its K, h = 1.65 m, the KITTI depth ground truth
for the crop and the LiDAR grid (built as in `eval_freespace.py`, visibility for
294 x 392). For a sample the server forces its real K, h and the outdoor preset,
and measures LiDAR grid metrics and depth AbsRel live for that request.
KITTI: CC BY-NC-SA 3.0, credited in `samples.json` and (M6) the README.

**Scale plausibility check (added after M4, `scale.py`).** In M4 the plane-fit
validity check (tilt + inlier share) accepted a crop of a building facade with
balconies (no road): tilt 7.4 deg, scale 3.45, about 4x the 0.87-0.89 of real
KITTI frames (recorded in the M4 commit's `report/api_check.json`).
`estimate_metric_scale` now also refuses when s is outside [0.5, 2.0], with the
reason "camera-height scale (N.Nx) disagrees with the network's own metres: the
detected floor is probably not the floor, or the camera height is wrong".
This check RELIES ON THE DEPTH NETWORK BEING ROUGHLY METRIC. It would wrongly
refuse valid floors with a relative-depth model, or wherever the metric model is
off by more than 2x — possibly indoors, since the network is Metric-OUTDOOR and
fine-tuned on KITTI. Phase 6 (NYU / DIODE) must report how often it refuses.

**M6 presentation rules for the verdict:**
- Present it as an estimate from ONE photo, not a guarantee: small input changes
  can flip it (M4: KITTI frame 5 as a sample said "Clear 6 -> 20 m"; the same frame
  uploaded with an estimated K said "Stop: car 15.8 m ahead"). Show the grid and
  the photo next to it so the user can see why.
- Do NOT suppress isolated OCCUPIED cells (no despeckling, no minimum blob size)
  to make the verdict steadier: an isolated cell may be a real thin obstacle
  (a pole, a bollard), and hiding it trades safety for a calmer display.

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
`GET /health`; `GET /samples` (the bundled KITTI gallery frames); `POST /predict`
(multipart: a sample id OR an uploaded image) returning JSON: status, refusal reason,
scale, tilt, FOV used and its source, timings, depth and painted photo as PNG
(base64), the grid as base64 uint8 + shape + cell size, and the path in metres
(the frontend draws the grid on a canvas). Input validation and size caps. CORS for
localhost and the Vercel domain.
Check: KITTI frame with h = 1.65 and hfov = 80 deg gives a sensible grid; a photo of
a wall returns a refusal with its reason; oversized and non-image uploads are rejected.

**M5. Docker + hosting (Azure Container Apps).**
`python:3.12-slim`, onnxruntime, numpy, opencv-python-headless, pillow, fastapi,
uvicorn, huggingface_hub. Run locally first, then deploy.
Check: public `/health` responds; `/predict` works on the KITTI test image; image size
is reported; cold start and per-request latency (median, p90 over the 6 samples,
warm-up excluded) measured on the deployed service -> `report/space_check.json`
(the numbers the UI displays); hardware reported.

*M5 status (paused, waiting for the owner's Azure for Students account):*
- DONE — model repo `BhaavinJain/monocular-freespace-models` published (public,
  commit `d272633e`): `depth.onnx`, `segformer.onnx`, their JSONs, the 6-frame gallery,
  model card with licences (`space/MODEL_CARD.md`; everything is non-commercial:
  SegFormer NVIDIA licence, KITTI and Virtual KITTI 2 CC BY-NC-SA 3.0).
- DONE — local image (`docker build -f space/Dockerfile -t freespace:local .` from the
  root): runs as uid 1000, no torch; 284 MB compressed, ~630 MB unpacked; `/health`
  ready 2-3.5 s after `docker run`; memory 261 MB idle, **633 MB after a 12 MP upload**,
  so the host needs >= 1 GB. `backend/check_api.py` against the container: all checks
  pass (`report/docker/api_check.json`). It caught a Linux-only bug (int64 labels vs
  opencv-python-headless), fixed in `pipeline.py`.
- BLOCKED — Hugging Face Space: `deploy.py --space` got 402, "hosting Gradio and Docker
  Spaces on free cpu-basic requires a PRO subscription". No Space was created.
  `space/README.md` (Space card) and `deploy.py --space` are now unused.
- TODO (Azure): push the image to ghcr.io; create the Container App (scale-to-zero,
  port 7860, >= 1 GB memory, `ORT_THREADS` = the vCPUs given, `CORS_ORIGINS` for M6);
  adapt `space/space_check.py` (it uses HF's `restart_space` / `space_info`: cold start
  must instead be measured from scale-to-zero, which on Azure includes pulling the
  284 MB image); run `check_api.py` and the latency check against the public URL.

**M6. React + Vite frontend (`web/`), on Vercel.**
Default view: the KITTI sample gallery (measured results). Upload mode, labelled
"experimental" (see "decisions after M3"): camera height (default 1.5 m, with a note that it sets the scale); FOV
(pre-filled from EXIF when available, editable); drivable mode; preset. Panels:
photo with the grid painted on it, depth map, BEV grid on a canvas with the path,
camera marker and distance lines (not rings, see "M4 definitions"). Show: measured
processing time, refusal reason when the plane fit fails, a "waking up" state for
cold starts (Azure scale-to-zero), and the
assumptions (camera height sets scale, FOV source, flat floor, trained on German
streets). API URL from `VITE_API_URL`.
Check: works locally against the local API, then on Vercel against the Azure service.

**M7. Polish (optional, only after M6 works end to end).**
Mobile layout, webcam capture, `/predict/sequence`.

## Useful commands (existing evaluation)

```
python validate_projection.py --frame 10 --depth_root data\kitti_depth
python check_ground_plane.py --frames 10 50 100
python eval_depth.py --depth_root data\kitti_depth --tag finetuned --weights checkpoints\depth_best.pt
python eval_freespace.py --depth_root data\kitti_depth --weights checkpoints\depth_best.pt --drivable road+sidewalk --tag finetuned_road_sidewalk
```