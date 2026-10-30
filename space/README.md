---
title: Monocular Free-Space Perception
emoji: 🛣️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: other
models:
  - HF_USER/monocular-freespace-models
short_description: One photo in, metric occupancy grid and path out (API)
---

# Monocular free-space perception — API

The backend of a portfolio project: one RGB photo in; depth, a metric top-down
occupancy grid (free / occupied / unknown), a planned path and a plain verdict out.
No LiDAR at run time. Code and all measurements:
[github.com/BhaavinJain/Monocular-Free-Space-Perception](https://github.com/BhaavinJain/Monocular-Free-Space-Perception).

- `GET /health`
- `GET /samples` — the KITTI gallery (measured results: real calibration, h = 1.65 m)
- `POST /predict` — multipart: `sample_id` or `file`, plus `camera_height`, `hfov_deg`,
  `drivable`, `preset`

Uploaded photos are **experimental**: camera height and field of view are
user-supplied, and accuracy on cameras other than KITTI's has not been measured.

Models and gallery: [HF_USER/monocular-freespace-models](https://huggingface.co/HF_USER/monocular-freespace-models).
**Non-commercial use only** (SegFormer licence, KITTI / Virtual KITTI 2 data); see that card.
