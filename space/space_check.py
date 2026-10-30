"""
Measure the deployed Space itself — these are the numbers the UI displays.

    python space/space_check.py

  hardware      what the Space actually runs on (HfApi().space_info)
  cold start    restart the Space (no rebuild; models are baked into the image),
                time from the restart request to the first /health 200, then the
                first /predict after it (the first inference is always slower)
  latency       1 warm-up request (excluded), then --rounds x the 6 gallery samples:
                median and p90 of the SERVER's own processing time per stage
                (timings_ms) and of the round trip from this machine (adds the
                network, so it depends on where the user is)
  uploads       --uploads requests with a 12 MP JPEG (a KITTI crop enlarged to
                4032 x 3024): upload mode decodes and shrinks a big photo, which
                the samples never do
  image size    of the local build of the same Dockerfile (docker image inspect)

-> report/space_check.json
"""
import argparse
import io
import json
import os
import subprocess
import time

import cv2
import numpy as np
import requests
from huggingface_hub import HfApi
from PIL import Image

STAGES = ("decode_resize", "depth", "seg", "geometry", "encode", "total")


def stats(v):
    return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "n": len(v)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--uploads", type=int, default=3)
    p.add_argument("--image", default="freespace:local", help="local tag of the same Dockerfile")
    p.add_argument("--samples", default="backend/samples")
    p.add_argument("--report_dir", default="report")
    p.add_argument("--skip_restart", action="store_true")
    a = p.parse_args()
    api = HfApi()
    user = api.whoami()["name"]
    repo, url = f"{user}/monocular-freespace", f"https://{user.lower()}-monocular-freespace.hf.space"

    def predict(**kw):
        t = time.perf_counter()
        r = requests.post(f"{url}/predict", timeout=300, **kw)
        r.raise_for_status()
        return r.json(), 1000 * (time.perf_counter() - t)

    # ---- cold start ----
    cold = None
    if not a.skip_restart:
        t0 = time.perf_counter()
        api.restart_space(repo)
        stages, left_running = [], False
        while time.perf_counter() - t0 < 900:
            stage = api.space_info(repo).runtime.stage
            if not stages or stages[-1][0] != stage:
                stages.append((stage, round(time.perf_counter() - t0, 1)))
            left_running |= stage != "RUNNING"
            # don't let the OLD container's /health answer count as the restarted one
            if left_running and stage == "RUNNING":
                try:
                    if requests.get(f"{url}/health", timeout=5).status_code == 200:
                        break
                except requests.RequestException:
                    pass
            time.sleep(2)
        ready_s = time.perf_counter() - t0
        first, first_rt = predict(data={"sample_id": requests.get(f"{url}/samples", timeout=60).json()[0]["id"]})
        cold = {"restart_to_health_s": ready_s, "stages_seen": stages,
                "first_predict_round_trip_ms": first_rt, "first_predict_server_ms": first["timings_ms"],
                "note": "restart without rebuild; a wake from free-tier sleep may add HF queue time"}
        print(f"cold start: {ready_s:.1f} s to /health, first predict {first_rt:.0f} ms round trip")

    # ---- per-request latency on the gallery ----
    ids = [s["id"] for s in requests.get(f"{url}/samples", timeout=60).json()]
    predict(data={"sample_id": ids[0]})                                   # warm-up, excluded
    server, rt = {s: [] for s in STAGES}, []
    for _ in range(a.rounds):
        for sid in ids:
            res, ms = predict(data={"sample_id": sid})
            assert res["status"] == "ok", res["reason"]
            for s in STAGES:
                server[s].append(res["timings_ms"][s])
            rt.append(ms)
    samples = {"server_ms": {s: stats(v) for s, v in server.items()}, "round_trip_ms": stats(rt)}
    print(f"samples: server total median {samples['server_ms']['total']['median']:.0f} ms "
          f"(p90 {samples['server_ms']['total']['p90']:.0f}), round trip median {samples['round_trip_ms']['median']:.0f} ms")

    # ---- 12 MP uploads ----
    img = cv2.cvtColor(cv2.imread(os.path.join(a.samples, ids[0], "image.png")), cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(cv2.resize(img, (4032, 3024), interpolation=cv2.INTER_CUBIC)).save(buf, "JPEG", quality=90)
    up_server, up_rt, statuses = [], [], []
    for _ in range(a.uploads):
        res, ms = predict(files={"file": ("12mp.jpg", buf.getvalue(), "image/jpeg")},
                          data={"camera_height": 1.65, "hfov_deg": 38.3})
        up_server.append(res["timings_ms"]["total"]), up_rt.append(ms), statuses.append(res["status"])
    uploads = {"file_mb": round(len(buf.getvalue()) / 1e6, 2), "server_total_ms": stats(up_server),
               "round_trip_ms": stats(up_rt), "statuses": statuses}
    print(f"12 MP upload: server median {uploads['server_total_ms']['median']:.0f} ms, "
          f"round trip median {uploads['round_trip_ms']['median']:.0f} ms ({uploads['file_mb']} MB)")

    info = api.space_info(repo)
    try:
        size = int(subprocess.run(["docker", "image", "inspect", a.image, "--format", "{{.Size}}"],
                                  capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        size = None
    report = {"space": repo, "url": url, "space_commit": info.sha,
              "hardware": {"current": info.runtime.hardware, "requested": info.runtime.requested_hardware},
              "image_size_mb_local_build": None if size is None else round(size / 1e6, 1),
              "cold_start": cold, "samples": samples, "uploads_12mp": uploads,
              "health": requests.get(f"{url}/health", timeout=60).json(),
              "display": {"processing_time_s": round(samples["server_ms"]["total"]["median"] / 1000, 2),
                          "upload_processing_time_s": round(uploads["server_total_ms"]["median"] / 1000, 2),
                          "cold_start_s": None if cold is None else round(cold["restart_to_health_s"])}}
    os.makedirs(a.report_dir, exist_ok=True)
    path = os.path.join(a.report_dir, "space_check.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"hardware {report['hardware']}, image {report['image_size_mb_local_build']} MB -> {path}")


if __name__ == "__main__":
    main()
