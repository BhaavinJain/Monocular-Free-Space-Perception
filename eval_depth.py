import argparse
import json
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from depth_model import DepthModel, METRIC_CKPT
from kitti_loader import KittiRawDrive
from metrics import compute_depth_metrics, range_stratified, average_metrics, valid_mask
from pipeline import fit_to_size, onnx_fn
from scale import estimate_metric_scale
 
KITTI_CAMERA_HEIGHT = 1.65
CROP_H, CROP_W = 350, 1218
 
 
def crop_bottom_centre(arr, K=None):
    """Crop to CROP_H x CROP_W from the bottom-centre; shift K to match."""
    H, W = arr.shape[:2]
    y0, x0 = H - CROP_H, (W - CROP_W) // 2
    out = arr[y0:y0 + CROP_H, x0:x0 + CROP_W]
    if K is None:
        return out
    Kc = K.copy()
    Kc[0, 2] -= x0          # pixel (x0, y0) is now (0, 0), so the
    Kc[1, 2] -= y0          # principal point moves by the same amount
    return out, Kc
 
 
def val_drives(root, depth_root):
    """Drives in the benchmark's val split whose raw images you have locally."""
    found, missing = [], []
    for name in sorted(os.listdir(os.path.join(depth_root, "val"))):
        date, num = name[:10], name.split("_drive_")[1][:4]
        if os.path.isdir(os.path.join(root, date, name)):
            found.append((date, num, name))
        else:
            missing.append(name)
    return found, missing
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--depth_root", required=True)
    p.add_argument("--checkpoint", default=METRIC_CKPT)
    p.add_argument("--weights", default=None, help="fine-tuned .pt (optional)")
    p.add_argument("--frame_step", type=int, default=10,
                   help="evaluate every Nth GT frame (1 = all, slow)")
    p.add_argument("--tag", default="pretrained")
    p.add_argument("--resize", type=float, default=1.0,
                   help="shrink (<1) or enlarge (>1) the photo before the network: "
                        "simulates a camera with a different focal length")
    p.add_argument("--report_dir", default="report")
    p.add_argument("--error_images", type=int, default=3,
                   help="save error pictures of the N frames contributing most 0-5 m error")
    p.add_argument("--size", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="the SHIPPING path: pipeline.fit_to_size to H x W (as the server does); "
                        "the scale is fitted at that size, metrics at the crop's full resolution")
    p.add_argument("--onnx_dir", default=None, help="use depth.onnx from here (the server's model) instead of PyTorch")
    a = p.parse_args()
    if a.onnx_dir and not a.size:
        p.error("--onnx_dir needs --size")

    if a.onnx_dir:
        with open(os.path.join(a.onnx_dir, "depth.json")) as fh:
            meta = json.load(fh)
        import onnxruntime as ort
        predict = onnx_fn(ort.InferenceSession(os.path.join(a.onnx_dir, "depth.onnx"),
                                               providers=["CPUExecutionProvider"]), meta["mean"], meta["std"])
    else:
        model = DepthModel(a.checkpoint, weights=a.weights)
        predict = model.predict
    if not a.onnx_dir and model.output_type != "metric":
        raise SystemExit("This evaluation needs a depth-up-to-scale model. "
                         "Relative Depth Anything outputs inverse depth with an "
                         "unknown shift — see explore_depth.py.")
 
    drives, missing = val_drives(a.root, a.depth_root)
    print(f"val drives with raw data: {[d[2] for d in drives]}")
    print(f"val drives without raw data (skipped): {len(missing)}")
    if not drives:
        raise SystemExit("No val drive has raw images under --root.")
 
    per = {"raw": [], "height": [], "oracle": []}
    scales, ratios, refused = [], [], []
    pooled_pred, pooled_raw, pooled_gt = [], [], []   # for the range-stratified tables
    worst = []          # (total near error, AbsRel, n px, name, frame, img, scaled pred, gt)
 
    for date, num, name in drives:
        drive = KittiRawDrive(a.root, date, num, depth_root=a.depth_root)
        gt_dir = os.path.join(a.depth_root, "val", name, "proj_depth",
                              "groundtruth", "image_02")
        frames = sorted(int(f[:-4]) for f in os.listdir(gt_dir))[::a.frame_step]
 
        for f in frames:
            if a.size:
                # exactly the server's input: bottom-anchored crop + resize, K follows;
                # the GT is cropped to the same box and the prediction brought back to it
                full = drive.load_image(f)
                img_m, K_fit, (y0, x0, ch, cw) = fit_to_size(full, drive.K, *a.size)
                img, gt = full[y0:y0 + ch, x0:x0 + cw], drive.load_depth_gt(f)[y0:y0 + ch, x0:x0 + cw]
                pred_fit = predict(img_m)
                pred = cv2.resize(pred_fit, (cw, ch), interpolation=cv2.INTER_LINEAR)
            else:
                img, K = crop_bottom_centre(drive.load_image(f), drive.K)
                gt = crop_bottom_centre(drive.load_depth_gt(f))
                if a.resize == 1.0:
                    pred = predict(img)
                else:
                    h, w = img.shape[:2]
                    small = cv2.resize(img, (round(w * a.resize), round(h * a.resize)),
                                       interpolation=cv2.INTER_AREA)
                    pred = cv2.resize(predict(small), (w, h),
                                      interpolation=cv2.INTER_LINEAR)
                pred_fit, K_fit = pred, K
            m = valid_mask(gt)
 
            per["raw"].append(compute_depth_metrics(pred, gt))
 
            s_oracle = np.median(gt[m]) / np.median(pred[m])
            per["oracle"].append(compute_depth_metrics(pred * s_oracle, gt))
 
            fit = estimate_metric_scale(pred_fit, K_fit, KITTI_CAMERA_HEIGHT)
            if not fit["valid"]:
                refused.append((name, f, fit["reason"]))
                continue
            s = fit["scale"]
            per["height"].append(compute_depth_metrics(pred * s, gt))
            scales.append(s)
            ratios.append(s / s_oracle)
            pooled_pred.append((pred * s)[m]); pooled_raw.append(pred[m]); pooled_gt.append(gt[m])
 
            # Keep the N frames contributing the MOST near-range (<= 5 m) error in
            # total: AbsRel x pixel count. Ranking by the average alone would pick
            # frames with only a handful of near pixels.
            near = m & (gt <= 5)
            if near.any():
                rel = np.abs(pred[near] * s - gt[near]) / gt[near]
                worst.append((rel.sum(), rel.mean(), int(near.sum()), name, f, img, pred * s, gt))
                worst.sort(key=lambda w: -w[0])
                del worst[a.error_images:]
        print(f"  {name}: {len(frames)} frames done")
 
    # ---------------- report ----------------
    report = {"tag": a.tag, "checkpoint": a.checkpoint, "weights": a.weights,
              "resize": a.resize,
              "frames_total": len(per["raw"]), "frames_refused": len(refused),
              "refused": refused[:20]}
    if a.size:
        report.update(size=a.size, backend="onnx" if a.onnx_dir else "pytorch",
                      worst_near_frames=[{"drive": w[3], "frame": w[4], "near_abs_rel": w[1], "near_pixels": w[2],
                                          "image": f"depth_error_{a.tag}_{w[3][17:21]}_{w[4]:03d}.png"} for w in worst])
    print(f"\nframes: {len(per['raw'])}, plane fit refused on {len(refused)}")
 
    cols = ["abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3"]
    print(f"\n{'mode':8s}" + "".join(f"{c:>10s}" for c in cols))
    for mode in ("raw", "height", "oracle"):
        if not per[mode]:
            continue
        avg = average_metrics(per[mode])
        report[mode] = avg
        print(f"{mode:8s}" + "".join(f"{avg[c]:10.4f}" for c in cols))
 
    if scales:
        report["scale"] = {"median_s": float(np.median(scales)),
                           "median_s_over_oracle": float(np.median(ratios)),
                           "p90_abs_scale_error": float(np.percentile(np.abs(np.array(ratios) - 1), 90)),
                           # share of frames whose camera-height scale is within 6% of the oracle
                           "frac_within_6pct": float(np.mean(np.abs(np.array(ratios) - 1) <= 0.06))}
        print(f"\nheight-based scale s: median {np.median(scales):.3f} "
              f"(1.0 = network already in correct metres)")
        print(f"s / s_oracle: median {np.median(ratios):.3f}, "
              f"90% of frames within {100 * report['scale']['p90_abs_scale_error']:.1f}% of oracle")
 
        pp, pr, pg = (np.concatenate(x) for x in (pooled_pred, pooled_raw, pooled_gt))
        bands_h, bands_r = range_stratified(pp, pg), range_stratified(pr, pg)
        report["range_stratified_height"] = bands_h
        report["range_stratified_raw"] = bands_r
        print("\nrange-stratified          height mode          |      raw mode")
        for band in bands_h:
            h, r = bands_h[band], bands_r[band]
            if h is None:
                print(f"  {band:>8s}: no GT pixels")
            else:
                print(f"  {band:>8s}: abs_rel {h['abs_rel']:.3f}  rmse {h['rmse']:5.2f} m  |"
                      f"  abs_rel {r['abs_rel']:.3f}  rmse {r['rmse']:5.2f} m  ({h['n']} px)")
 
    # Error pictures: where are the near-range errors, and what are they on?
    os.makedirs(os.path.join(a.report_dir, "image"), exist_ok=True)
    if worst:
        print(f"\nframes contributing most error on pixels <= 5 m (height mode):")
    for _, near_err, n_near, name, f, img, pred_s, gt in worst:
        m = valid_mask(gt)
        v, u = np.nonzero(m)
        err = np.abs(pred_s[m] - gt[m]) / gt[m]
        near = gt[m] <= 5
        fig, ax = plt.subplots(2, 1, figsize=(img.shape[1] / 100, 2 * img.shape[0] / 100 + 0.6))
        for k, (sel, title) in enumerate([
                (slice(None), f"{name} frame {f}: all GT pixels, coloured by error (green 0% -> red >= 50%)"),
                (near, f"only pixels <= 5 m: AbsRel {near_err:.3f} over {near.sum()} px")]):
            ax[k].imshow(img)
            ax[k].scatter(u[sel], v[sel], c=err[sel], s=2, cmap="RdYlGn_r", vmin=0, vmax=0.5)
            ax[k].set_title(title)
            ax[k].axis("off")
        fig.tight_layout()
        path = os.path.join(a.report_dir, "image", f"depth_error_{a.tag}_{name[17:21]}_{f:03d}.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"  frame {f}: near AbsRel {near_err:.3f} over {n_near} px -> {path}")
 
    path = os.path.join(a.report_dir, f"depth_eval_{a.tag}.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nreport -> {path}")
 
 
if __name__ == "__main__":
    main()