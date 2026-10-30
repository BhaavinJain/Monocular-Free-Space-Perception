---
license: other
license_name: non-commercial-mixed
base_model: depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf
tags:
  - onnx
  - depth-estimation
  - semantic-segmentation
  - kitti
---

# Monocular free-space perception — ONNX models and KITTI gallery

Files served by the Space [HF_USER/monocular-freespace](https://huggingface.co/spaces/HF_USER/monocular-freespace):
one RGB photo in, a metric top-down occupancy grid and a planned path out.
Code, evaluation scripts and every number below:
[github.com/BhaavinJain/Monocular-Free-Space-Perception](https://github.com/BhaavinJain/Monocular-Free-Space-Perception).

**Non-commercial use only.** Every component below carries a non-commercial
restriction (or derives from data that does). Use for research, evaluation or
education.

## Licences

| Component | Licence | Source |
|---|---|---|
| Depth Anything V2 **Small** (the base architecture/weights) | Apache-2.0 | [Depth-Anything-V2 README](https://github.com/DepthAnything/Depth-Anything-V2) |
| Depth Anything V2 **Metric-Outdoor-Small** (the starting checkpoint) | **No licence stated** on its model card or in the repository. It was trained on Virtual KITTI 2, which is CC BY-NC-SA 3.0, non-commercial only | [metric_depth README](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth), [Virtual KITTI 2 terms](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) |
| `depth.onnx` (**fine-tuned by this project on KITTI**) | derived from the above and from KITTI (CC BY-NC-SA 3.0): treat as non-commercial | — |
| `segformer.onnx` (nvidia/segformer-b0-finetuned-cityscapes-1024-1024, unmodified) | NVIDIA Source Code License for SegFormer: "only may be used … non-commercially", i.e. research or evaluation only | [NVlabs/SegFormer LICENSE](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) |
| `samples/` (6 KITTI frames, depth ground truth, LiDAR grids) | KITTI: CC BY-NC-SA 3.0 | [cvlibs.net/datasets/kitti](https://www.cvlibs.net/datasets/kitti/) |

This table states what the sources say; it is not legal advice.

## Files

| File | What |
|---|---|
| `depth.onnx` | Depth Anything V2 Metric-Outdoor-Small, fine-tuned on KITTI raw (`finetune_depth.py`), exported by `export_onnx.py`. Input `pixel_values` (1, 3, H, W) float32, H and W **multiples of 14**, dynamic. Output depth (1, H, W) in metres. |
| `segformer.onnx` | SegFormer-B0 Cityscapes with the upsampling to input size and the argmax inside the graph. Output labels (1, H, W), Cityscapes train ids. |
| `*.json` | normalisation (`(RGB / 255 - mean) / std`), input/output description, class names |
| `samples/` | the Space's gallery: 375 x 500 crops of KITTI drive 2011_09_26_0023 (val) with calibration, depth ground truth and LiDAR occupancy grids (`backend/make_samples.py`) |

Export note: Depth Anything's head (transformers 5.18.0, `modeling_depth_anything.py`
line 298) calls `int()` on the image size, which freezes the output size during
export; `export_onnx.py` patches that line for export only, and checks that the
patch leaves PyTorch's output bit-identical.

## Measured (KITTI only — not the accuracy on other cameras)

- ONNX vs PyTorch on 20 KITTI val frames: depth median relative difference 0.0000 %,
  max 0.0065 %; segmentation labels 100.000 % identical (`report/onnx_parity.json`).
- Fine-tuned depth on KITTI drive 0023 (val, 47 frames, camera-height scale):
  AbsRel 0.071 (`report/depth_eval_finetuned.json`).

These numbers are for the KITTI camera at 1.65 m. Accuracy on other cameras,
heights and scenes has **not been measured**.

## Citation

Geiger, Lenz, Stiller, Urtasun. *Vision meets Robotics: The KITTI Dataset.* IJRR 2013.
Yang et al. *Depth Anything V2.* NeurIPS 2024. Xie et al. *SegFormer.* NeurIPS 2021.
