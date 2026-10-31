"""
Generate web/src/results.json (and copy the failure-gallery images) from report/.

    python web/make_results.py            (from the project root)

The About page shows ONLY numbers from this file, and this file only copies
numbers out of report/*.json — none are typed here (hard rule 5). Every block
says what it was measured on. Captions are words; their numbers come from the
reports too. If a rerun changes which frames are the worst, this script stops
instead of putting an old caption under a new picture.
"""
import argparse
import json
import os
import shutil

# what each failure picture shows, in words (keyed by frame: see the docstring)
CAPTIONS = {
    195: "The far ends of the parked cars, near the far edge of the map, were called free (yellow): "
         "far away, object outlines blur into the road.",
    115: "The side of a parked car on the left, far ahead, slipped into the free area: "
         "the errors sit at the edges of objects.",
    465: "Two mistakes at once: car and hedge edges far ahead called free (yellow), and a strip of "
         "open road on the right wrongly marked blocked (orange where the laser saw free) - the safe kind of mistake.",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="report")
    p.add_argument("--out", default="web/src/results.json")
    p.add_argument("--images", default="web/public/failures")
    a = p.parse_args()
    L = lambda name: json.load(open(os.path.join(a.report, name)))

    d_ship, d_pre = L("depth_eval_ship_finetuned_onnx.json"), L("depth_eval_ship_pretrained_pytorch.json")
    # the M6 regression run equals depth_eval_finetuned.json in every old key and adds
    # frac_within_6pct, which the Phase 2 report predates
    d_full, d_full_pre = L("depth_eval_m6_regression.json"), L("depth_eval_pretrained.json")
    f_ship, f_geom = L("freespace_ship_road_sidewalk_onnx.json"), L("freespace_ship_geometry_onnx.json")
    f_full = L("freespace_finetuned_road_sidewalk.json")
    space, azure = L("space_check.json"), L("azure/api_check.json")
    web = L("web_timing.json") if os.path.exists(os.path.join(a.report, "web_timing.json")) else None
    docker = L("docker/api_check.json")

    abs_rel = d_ship["height"]["abs_rel"]
    m, mf = f_ship["metrics"], f_full["metrics"]
    crops = next(c for c in azure["checks"] if "crops" in c)["crops"]
    phones = docker.get("phone_photos") or {}
    out = {
        "measured_on": {"frames": d_ship["frames_total"], "drive": "KITTI drive 2011_09_26_0023 (validation, not used for training)",
                        "camera": "KITTI's front colour camera, mounted 1.65 m above the road",
                        "input": f"{d_ship['size'][0]} x {d_ship['size'][1]} px, a 4:3 crop of the camera image - what the app uses",
                        "models": "ONNX Runtime, the same models the server runs"},
        "depth": {
            "abs_rel": abs_rel, "pct": 100 * abs_rel, "example_10m_m": 10 * abs_rel,
            "frames": d_ship["frames_total"], "refused": d_ship["frames_refused"],
            "pretrained_abs_rel": d_pre["height"]["abs_rel"], "pretrained_backend": "PyTorch",
            "improvement_pct": 100 * (1 - abs_rel / d_pre["height"]["abs_rel"]),
            "scale_within_6pct": d_ship["scale"]["frac_within_6pct"],
            "full_res": {"abs_rel": d_full["height"]["abs_rel"], "pretrained_abs_rel": d_full_pre["height"]["abs_rel"],
                         "scale_within_6pct": d_full["scale"]["frac_within_6pct"],
                         "input": "350 x 1218 px, the full-width crop (Phase 2)"},
        },
        "freespace": {
            "iou": m["all"]["free_iou"], "false_free": m["all"]["false_free_rate"],
            "false_free_margin": m["all"]["false_free_rate_inflated"], "missed": m["all"]["missed_obstacle_rate"],
            "cells": m["all"]["cells_evaluated"], "robot_radius_m": f_ship["robot_radius_m"],
            "bands": [{"band": k, "false_free": v["false_free_rate"], "false_free_margin": v["false_free_rate_inflated"]}
                      for k, v in m.items() if k != "all"],
            "geometry_only": {"iou": f_geom["metrics"]["all"]["free_iou"],
                              "false_free": f_geom["metrics"]["all"]["false_free_rate"],
                              "missed": f_geom["metrics"]["all"]["missed_obstacle_rate"]},
            "full_res": {"iou": mf["all"]["free_iou"], "false_free": mf["all"]["false_free_rate"],
                         "false_free_margin": mf["all"]["false_free_rate_inflated"],
                         "missed": mf["all"]["missed_obstacle_rate"], "cells": mf["all"]["cells_evaluated"],
                         "input": "350 x 1218 px, the full-width crop (Phase 4)"},
        },
        "timing": {"server_s": space["samples"]["server_ms"]["total"]["median"] / 1000,
                   "server_p90_s": space["samples"]["server_ms"]["total"]["p90"] / 1000,
                   "upload_server_s": space["uploads_12mp"]["server_total_ms"]["median"] / 1000,
                   "cold_start_s": space["cold_start"]["zero_to_health_s"]["median"],
                   "requests": space["samples"]["server_ms"]["total"]["n"],
                   "hardware": f"Azure Container Apps, {space['hardware']['vcpu']:g} vCPU, {space['region']}",
                   "browser": web},
        "refusal": {"refused": sum(c["status"] == "refused" for c in crops), "total": len(crops),
                    "what": "crops of the upper part of KITTI images - trees, sky, a building, a van - with no road in them"},
        "indoor": [{"status": v["status"], "reason": v["reason"], "camera_height_m": v["camera"]["height_m"],
                    "what": "a phone photo of an indoor corridor (not published: it is the owner's)"}
                   for v in phones.values()][:1],
        "failures": [],
    }
    os.makedirs(a.images, exist_ok=True)
    for w in f_ship["worst_frames"]:
        if w["frame"] not in CAPTIONS:
            raise SystemExit(f"no caption for frame {w['frame']}: look at {w['image']} and write one")
        shutil.copy(os.path.join(a.report, "image", w["image"]), os.path.join(a.images, w["image"]))
        out["failures"].append({"image": f"failures/{w['image']}", "frame": w["frame"],
                                "false_free_cells": w["false_free_cells"], "caption": CAPTIONS[w["frame"]]})
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"{a.out}: depth {out['depth']['pct']:.1f}% | free-space agreement {100 * out['freespace']['iou']:.0f}% | "
          f"{len(out['failures'])} failure images | browser timing {'yes' if web else 'not yet measured'}")


if __name__ == "__main__":
    main()
