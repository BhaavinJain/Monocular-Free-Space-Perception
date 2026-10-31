"""
Export the fine-tuned depth model and SegFormer-B0 to ONNX, then prove the
ONNX files compute the same thing as PyTorch.

Both graphs take a normalised (1, 3, H, W) float32 image with DYNAMIC H and W
(depth needs multiples of 14). Normalisation stays outside the graph; its
mean/std go in a JSON next to each model, so the server (numpy only) does it.

The risk with a dynamic export is DINOv2's position embeddings: they are
learned on a 37x37 patch grid and bicubically resized to (H/14, W/14) for
each input. If the exporter froze that target size as a constant, the model
would still RUN at other sizes but with wrong embeddings — silently worse
depth. So parity is also checked at a second, very different shape.

The position embeddings turned out fine; the depth HEAD was the problem (see
dynamic_head_forward), so it is patched here, for export only.

Checks (all against PyTorch on CPU, saved to report/onnx_parity.json):
  0. the head patch changes nothing in PyTorch (bit-identical depth)
  1. model parity on N KITTI val frames: depth relative difference, label agreement
  2. the same at two other shapes, 518 x 392 and 476 x 630 (dynamic-shape test)
  3. end to end: pipeline.run() with PyTorch vs ONNX depth_fn/seg_fn —
     scale difference and % of grid cells with the same state
"""
import argparse
import json
import os

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
import transformers
from transformers.models.depth_anything import modeling_depth_anything as mda

from models.depth_model import METRIC_CKPT, DepthModel, to_input_tensor
from evaluation.eval_depth import crop_bottom_centre
from evaluation.eval_freespace import KITTI_CAMERA_HEIGHT
from freespace.kitti_loader import KittiRawDrive
from freespace.pipeline import onnx_fn, run
from models.segmenter import SEG_CKPT, Segmenter

EXTRA_HW = [(518, 392), (476, 630)]     # multiples of 14; portrait and 4:3-ish, unlike KITTI


def dynamic_head_forward(self, hidden_states, patch_height, patch_width):
    """
    Copy of DepthAnythingDepthEstimationHead.forward from transformers 5.18.0
    (modeling_depth_anything.py, line 298), minus two int() calls. During
    export, int() turns the symbolic image size into the example's numbers,
    so the last upsample was frozen at 350 x 1218 and every other input size
    came back stretched to that — with no error. Without int() the size stays
    tied to the input. In PyTorch the values are the same (patch counts are
    already ints); check 0 proves it.
    """
    hidden_states = hidden_states[self.head_in_index]
    predicted_depth = self.conv1(hidden_states)
    predicted_depth = F.interpolate(predicted_depth,
                                    (patch_height * self.patch_size, patch_width * self.patch_size),
                                    mode="bilinear", align_corners=True)
    predicted_depth = self.conv2(predicted_depth)
    predicted_depth = self.activation1(predicted_depth)
    predicted_depth = self.conv3(predicted_depth)
    predicted_depth = self.activation2(predicted_depth) * self.max_depth
    return predicted_depth.squeeze(dim=1)


class DepthNet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):                      # (1, 3, H, W) -> (1, H, W) metres
        return self.model(pixel_values=x).predicted_depth


class SegNet(torch.nn.Module):
    """Upsample + argmax inside the graph, so ONNX returns exactly what
    Segmenter.predict returns and the server needs no extra code."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):                      # (1, 3, H, W) -> (1, H, W) class ids
        logits = self.model(pixel_values=x).logits
        logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        return logits.argmax(1)


def export(net, x, path, output_name):
    dim = torch.export.Dim.DYNAMIC
    torch.onnx.export(net, (x,), path, dynamo=True, external_data=False,
                      input_names=["pixel_values"], output_names=[output_name],
                      dynamic_shapes=({2: dim, 3: dim},))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--depth_root", default="data/kitti_depth")
    p.add_argument("--drive", default="0023")
    p.add_argument("--weights", default="checkpoints/depth_best.pt")
    p.add_argument("--out_dir", default="checkpoints/onnx")
    p.add_argument("--report_dir", default="report")
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--skip_export", action="store_true", help="reuse existing .onnx files")
    a = p.parse_args()
    torch.set_grad_enabled(False)

    drive = KittiRawDrive(a.root, "2011_09_26", a.drive, depth_root=a.depth_root)
    gt_dir = os.path.join(a.depth_root, "val", drive.drive_name, "proj_depth", "groundtruth", "image_02")
    gt_frames = sorted(int(f[:-4]) for f in os.listdir(gt_dir))
    frames = [gt_frames[i] for i in np.linspace(0, len(gt_frames) - 1, a.frames).astype(int)]

    # PyTorch on CPU: the reference must not carry GPU (TF32 / cuDNN) differences of its own
    depth_model, seg = DepthModel(weights=a.weights, device="cpu"), Segmenter(device="cpu")
    nets = {"depth": DepthNet(depth_model.model).eval(), "segformer": SegNet(seg.model).eval()}
    norm = {"depth": (depth_model.mean, depth_model.std), "segformer": (seg.mean, seg.std)}
    paths = {k: os.path.join(a.out_dir, f"{k}.onnx") for k in nets}

    img0, _ = crop_bottom_centre(drive.load_image(frames[0]), drive.K)

    # ---- 0. the patch must not change PyTorch's output (KITTI size + the extra shapes) ----
    patch_identical = {}
    shapes = [img0.shape[:2]] + EXTRA_HW
    xs = [to_input_tensor(cv2.resize(img0, (w, h), interpolation=cv2.INTER_AREA), *norm["depth"]).unsqueeze(0)
          for h, w in shapes]
    before = [nets["depth"](x) for x in xs]
    mda.DepthAnythingDepthEstimationHead.forward = dynamic_head_forward
    for (h, w), x, b in zip(shapes, xs, before):
        patch_identical[f"{h}x{w}"] = bool(torch.equal(nets["depth"](x), b))
    print("patch leaves PyTorch depth identical:", patch_identical)

    x0 = xs[0]
    if not a.skip_export:
        os.makedirs(a.out_dir, exist_ok=True)
        export(nets["depth"], x0, paths["depth"], "depth")
        export(nets["segformer"], to_input_tensor(img0, *norm["segformer"]).unsqueeze(0),
               paths["segformer"], "labels")
        meta = {"depth": {"checkpoint": METRIC_CKPT, "weights": a.weights,
                          "output": "depth (1, H, W) float32, metres; H and W must be multiples of 14"},
                "segformer": {"checkpoint": SEG_CKPT, "labels": seg.names,
                              "output": "labels (1, H, W) int64, Cityscapes train ids"}}
        for k in nets:
            mean, std = norm[k]
            meta[k].update(input="pixel_values (1, 3, H, W) float32 = (RGB / 255 - mean) / std",
                           mean=mean.flatten().tolist(), std=std.flatten().tolist(),
                           export_hw=list(img0.shape[:2]))
            with open(os.path.join(a.out_dir, f"{k}.json"), "w") as fh:
                json.dump(meta[k], fh, indent=2)
            print(f"exported {paths[k]} ({os.path.getsize(paths[k]) / 1e6:.1f} MB)")

    sess = {k: ort.InferenceSession(paths[k], providers=["CPUExecutionProvider"]) for k in nets}
    mean_std = {k: (v[0].numpy().reshape(1, 1, 3).astype(np.float64),
                    v[1].numpy().reshape(1, 1, 3).astype(np.float64)) for k, v in norm.items()}

    def model_parity(img):
        """Same normalised tensor into both runtimes -> (torch depth, onnx depth, torch labels, onnx labels)."""
        xd = to_input_tensor(img, *norm["depth"]).unsqueeze(0)
        xs = to_input_tensor(img, *norm["segformer"]).unsqueeze(0)
        return (nets["depth"](xd)[0].numpy(), sess["depth"].run(None, {"pixel_values": xd.numpy()})[0][0],
                nets["segformer"](xs)[0].numpy(), sess["segformer"].run(None, {"pixel_values": xs.numpy()})[0][0])

    def depth_stats(rel):
        return {"median_rel_diff": float(np.median(rel)), "p99_9_rel_diff": float(np.percentile(rel, 99.9)),
                "max_rel_diff": float(rel.max())}

    # ---- 1. model parity on KITTI frames ----
    rels, agree, worst, per_frame = [], [], [], []
    for f in frames:
        img, _ = crop_bottom_centre(drive.load_image(f), drive.K)
        dt, do, lt, lo = model_parity(img)
        rel = np.abs(do - dt) / dt
        rels.append(rel.ravel())
        agree.append((lo == lt).ravel())
        for i in np.argsort(rel.ravel())[-5:]:          # 5 worst pixels per frame, kept for the report
            v, u = np.unravel_index(i, rel.shape)
            worst.append({"frame": f, "row": int(v), "col": int(u), "torch_m": float(dt[v, u]),
                          "onnx_m": float(do[v, u]), "rel_diff": float(rel[v, u])})
        per_frame.append({"frame": f, "max_rel_diff": float(rel.max()),
                          "label_agreement": float((lo == lt).mean())})
        print(f"  frame {f}: depth max rel {rel.max():.2e}, labels {100 * (lo == lt).mean():.3f}%")
    rel_all, agree_all = np.concatenate(rels), np.concatenate(agree)
    kitti = {"frames": frames, "image_hw": list(img0.shape[:2]), **depth_stats(rel_all),
             "label_agreement": float(agree_all.mean()),
             "min_frame_label_agreement": min(r["label_agreement"] for r in per_frame),
             "worst_pixels": sorted(worst, key=lambda w: -w["rel_diff"])[:10], "per_frame": per_frame}
    kitti["depth_pass"] = kitti["median_rel_diff"] < 0.005 and kitti["max_rel_diff"] < 0.02
    kitti["seg_pass"] = kitti["label_agreement"] > 0.995

    # ---- 2. dynamic-shape test at other sizes (one KITTI frame, resized) ----
    extra = {}
    for h, w in EXTRA_HW:
        dt, do, lt, lo = model_parity(cv2.resize(img0, (w, h), interpolation=cv2.INTER_AREA))
        if do.shape != dt.shape or lo.shape != lt.shape:      # a frozen size shows up here
            extra[f"{h}x{w}"] = {"depth_pass": False, "seg_pass": False,
                                 "error": f"output shapes onnx {do.shape}/{lo.shape} vs torch {dt.shape}/{lt.shape}"}
            continue
        r = {**depth_stats(np.abs(do - dt) / dt), "label_agreement": float((lo == lt).mean())}
        r["depth_pass"] = r["median_rel_diff"] < 0.005 and r["max_rel_diff"] < 0.02
        r["seg_pass"] = r["label_agreement"] > 0.995
        extra[f"{h}x{w}"] = r

    # ---- 3. end to end through pipeline.run ----
    fns = {"torch": (depth_model.predict, seg.predict),
           "onnx": (onnx_fn(sess["depth"], *mean_std["depth"]), onnx_fn(sess["segformer"], *mean_std["segformer"]))}
    e2e = []
    for f in frames:
        img, K = crop_bottom_centre(drive.load_image(f), drive.K)
        o = {k: run(img, K, KITTI_CAMERA_HEIGHT, d, s, drivable="road+sidewalk") for k, (d, s) in fns.items()}
        r = {"frame": f, "status": [o["torch"]["status"], o["onnx"]["status"]]}
        if r["status"] == ["ok", "ok"]:
            r.update(scale_rel_diff=abs(o["onnx"]["scale"] - o["torch"]["scale"]) / o["torch"]["scale"],
                     grid_agreement=float((o["onnx"]["grid"] == o["torch"]["grid"]).mean()),
                     inflated_agreement=float((o["onnx"]["grid_inflated"] == o["torch"]["grid_inflated"]).mean()))
        e2e.append(r)
    ok = [r for r in e2e if r["status"] == ["ok", "ok"]]
    end_to_end = {"drivable": "road+sidewalk", "frames_both_ok": len(ok),
                  "frames_status_differs": [r["frame"] for r in e2e if r["status"][0] != r["status"][1]],
                  "scale_rel_diff_median": float(np.median([r["scale_rel_diff"] for r in ok])),
                  "scale_rel_diff_max": float(max(r["scale_rel_diff"] for r in ok)),
                  "grid_agreement_mean": float(np.mean([r["grid_agreement"] for r in ok])),
                  "grid_agreement_min": min(r["grid_agreement"] for r in ok),
                  "inflated_agreement_mean": float(np.mean([r["inflated_agreement"] for r in ok])),
                  "inflated_agreement_min": min(r["inflated_agreement"] for r in ok),
                  "per_frame": e2e}

    report = {"versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                           "onnx": onnx.__version__, "onnxruntime": ort.__version__},
              "opset": {k: onnx.load(paths[k], load_external_data=False).opset_import[0].version for k in nets},
              "file_mb": {k: round(os.path.getsize(paths[k]) / 1e6, 1) for k in nets},
              "thresholds": {"depth_median_rel": 0.005, "depth_max_rel": 0.02, "label_agreement": 0.995},
              "patch_identical_in_torch": patch_identical,
              "kitti": kitti, "extra_shapes": extra, "end_to_end": end_to_end}
    report["passed"] = (all(patch_identical.values()) and
                        all(r["depth_pass"] and r["seg_pass"] for r in [kitti, *extra.values()]))
    path = os.path.join(a.report_dir, "onnx_parity.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)

    for name, r in (("kitti", kitti), *extra.items()):
        if "error" in r:
            print(f"{name:9s} FAIL: {r['error']}")
            continue
        print(f"{name:9s} depth median {100 * r['median_rel_diff']:.4f}%  p99.9 {100 * r['p99_9_rel_diff']:.4f}%  "
              f"max {100 * r['max_rel_diff']:.4f}%  [{'PASS' if r['depth_pass'] else 'FAIL'}]   "
              f"labels {100 * r['label_agreement']:.3f}%  [{'PASS' if r['seg_pass'] else 'FAIL'}]")
    e = end_to_end
    print(f"end-to-end ({e['frames_both_ok']}/{len(frames)} both ok): scale diff median "
          f"{100 * e['scale_rel_diff_median']:.4f}% max {100 * e['scale_rel_diff_max']:.4f}%  "
          f"grid agree mean {100 * e['grid_agreement_mean']:.2f}% min {100 * e['grid_agreement_min']:.2f}%  "
          f"inflated mean {100 * e['inflated_agreement_mean']:.2f}% min {100 * e['inflated_agreement_min']:.2f}%")
    print(f"\nmodel parity {'PASSED' if report['passed'] else 'FAILED'} -> {path}")


if __name__ == "__main__":
    main()
