""" Output of the actual depth network
If the output is depth up to a single scale factor, s = h / d , works. 
If it isn't, the scale-recovery method silently gives wrong metres
instead of trusting documentation, the script tests the claim against LiDAR ground truth.'

  (a) scale-only depth      gt   = s * pred           1 unknown
  (b) affine depth          gt   = s * pred + t       2 unknowns
  (c) affine inverse depth  1/gt = s * pred + t       2 unknowns (disparity)
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from depth_model import DepthModel, RELATIVE_CKPT, METRIC_CKPT
from kitti_loader import KittiRawDrive
from metrics import valid_mask

def fit_hypotheses(pred, gt):
    # pred and gt arrive as 1-D arrays: the network's value and the LiDAR depth at each valid pixel, in matching order
    res = {}    # Returning AbsRel and parameter

    # (a) scale-only depth      gt   = s * pred           1 unknown
    s = (pred @ gt) / (pred @ pred)
    res["a : scale-only depth"] = (np.mean(np.abs(s * pred - gt) / gt), s)

    # (b) affine depth          gt   = s * pred + t       2 unknowns
    A = np.stack([pred, np.ones_like(pred)], axis=1)
    s, t = np.linalg.lstsq(A, gt, rcond=None)[0]
    res["b : affine depth"] = (np.mean(np.abs(gt - (s * pred + t - gt)) / gt), (s, t))

    # (c) affine inverse depth  1/gt = s * pred + t , then depth = 1 / (s * pred + t)   2 unknowns (disparity)
    (s, t), *_ = np.linalg.lstsq(A, 1.0 / gt, rcond=None)
    inv = np.clip(s * pred + t, 1e-6, None)        # guard against <= 0
    res["c: affine inverse depth"] = (np.mean(np.abs(1.0 / inv - gt) / gt), (s, t))
    
    return res

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="23")
    p.add_argument("--depth_root", required=True)
    p.add_argument("--checkpoint", default="metric", choices=["metric", "relative"])
    p.add_argument("--frames", type=int, nargs="+", default=[10, 50, 100, 150])
    p.add_argument("--out", default="explore_depth.png")
    a = p.parse_args()
 
    ckpt = METRIC_CKPT if a.checkpoint == "metric" else RELATIVE_CKPT
    model = DepthModel(ckpt)
    drive = KittiRawDrive(a.root, a.date, a.drive, depth_root=a.depth_root)
    print(f"checkpoint: {ckpt}  (config says: {model.output_type})\n")

    # Per-frame loop
    totals, first = {}, None
    for f in a.frames:
        gt = drive.load_depth_gt(f)
        if gt is None:
            print(f"frame {f} has no LiDAR depth, skipping")
            continue
        pred = model.predict(drive.load_image(f))
        m = valid_mask(gt)
        res = fit_hypotheses(pred[m].astype(np.float64), gt[m].astype(np.float64))
        print(f"frame {f}: " + "  ".join(f"{k}={v[0]:.3f}" for k, v in res.items()))

        for k, v in res.items():
            totals.setdefault(k, []).append(v[0])
        if first is None:
            first = (pred[m], gt[m])

        if not totals:
            raise SystemExit("No frames had ground truth — check --depth_root and --frames.")

        print("\nMean AbsRel after the best fit of each hypothesis (lower = better):")

        # Verify that the first frame's results are representative of the rest
        a_err = np.mean(totals["a: scale-only depth"])
        b_err = np.mean(totals["b: affine depth"])
        c_err = np.mean(totals["c: affine inverse depth"])
        print()

        if c_err < 0.8 * min(a_err, b_err):
            print("VERDICT: affine INVERSE depth (disparity). One constraint is NOT "               
              "enough — do not use this checkpoint with s = h/d.")
        elif a_err <= 1.15 * b_err + 1e-3:
            print("VERDICT: depth up to a single SCALE. The shift adds little, so "       # Scale Only
              "camera-height scale recovery is valid for this checkpoint.")
        else:
            print("VERDICT: affine depth (scale + shift). One constraint is not "
              "enough; you'd need a second (e.g. two known distances).")

        # Scatter plot for the first frame (whichever wwe list first when passing --frames)
            pred_v, gt_v = first
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
            ax[0].scatter(pred_v, gt_v, s=1, alpha=0.1)
            ax[0].set_xlabel("network output"); ax[0].set_ylabel("LiDAR depth (m)")
            ax[0].set_title("depth vs output — straight line through 0 = scale-only")
            ax[1].scatter(pred_v, 1.0 / gt_v, s=1, alpha=0.1, color="tab:orange")
            ax[1].set_xlabel("network output"); ax[1].set_ylabel("1 / LiDAR depth (1/m)")
            ax[1].set_title("inverse depth vs output — straight line = disparity")
            fig.tight_layout() 
            fig.savefig(a.out, dpi=110)
            plt.close(fig)

if __name__ == "__main__":
    main()