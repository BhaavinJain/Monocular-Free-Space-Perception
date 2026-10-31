"""
Measure the deployed service itself (Azure Container Apps) — these are the
numbers the UI displays.

    python space/space_check.py

  hardware     what Azure gives the app (az containerapp show: vCPU, memory,
               region, scaling) and what the container sees (/health: CPU model,
               cgroup CPU limit)
  cold start   --cold_starts times: wait until the app has scaled to ZERO
               replicas (no traffic for the cooldown), then time a /health
               request — the ingress holds it while a replica is scheduled, the
               image pulled (if that node lacks it) and the models loaded —
               then time the first /predict after it separately
  latency      1 warm-up request (excluded), then --rounds x the 6 gallery
               samples: median and p90 of the SERVER's processing time per stage
               (timings_ms) and of the round trip from this machine (adds the
               network, so it depends on where the user is)
  uploads      --uploads requests with a 12 MP JPEG (a KITTI crop enlarged to
               4032 x 3024): upload mode decodes and shrinks a big photo
  image size   compressed, as stored in ghcr.io (what Azure pulls)

Only read-only az calls; nothing is created or changed. -> report/space_check.json
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np
import requests
from PIL import Image

STAGES = ("decode_resize", "depth", "seg", "geometry", "encode", "total")
AZ = shutil.which("az") or "az"             # az.cmd on Windows


def az(*args):
    return json.loads(subprocess.run([AZ, *args, "-o", "json"], capture_output=True, text=True, check=True).stdout)


def stats(v):
    return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "n": len(v)}


def compressed_size(ref):
    """Sum of the compressed layers in the registry manifest (linux/amd64)."""
    m = json.loads(subprocess.run(["docker", "manifest", "inspect", ref], capture_output=True, text=True, check=True).stdout)
    if "manifests" in m:                    # an index: pick the linux/amd64 image
        d = next(x["digest"] for x in m["manifests"] if x.get("platform", {}).get("architecture") == "amd64")
        m = json.loads(subprocess.run(["docker", "manifest", "inspect", f"{ref.split(':')[0]}@{d}"],
                                      capture_output=True, text=True, check=True).stdout)
    return m["config"]["size"] + sum(l["size"] for l in m["layers"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rg", default="rg-freespace")
    p.add_argument("--app", default="freespace-api")
    p.add_argument("--cold_starts", type=int, default=3)
    p.add_argument("--max_idle_wait_s", type=int, default=1800)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--uploads", type=int, default=3)
    p.add_argument("--samples", default="backend/samples")
    p.add_argument("--report_dir", default="report")
    a = p.parse_args()

    app = az("containerapp", "show", "-n", a.app, "-g", a.rg)
    url = f"https://{app['properties']['configuration']['ingress']['fqdn']}"
    container = app["properties"]["template"]["containers"][0]

    def replicas():
        return len(az("containerapp", "replica", "list", "-n", a.app, "-g", a.rg))

    def predict(**kw):
        t = time.perf_counter()
        r = requests.post(f"{url}/predict", timeout=600, **kw)
        r.raise_for_status()
        return r.json(), 1000 * (time.perf_counter() - t)

    # ---- cold starts from zero ----
    # Wall-clock times are recorded so a cold start can be checked afterwards for
    # overlap with any other traffic (someone opening the URL during the window
    # would start the replica and make the measured cold start look shorter).
    now = lambda: datetime.now().astimezone().isoformat(timespec="seconds")
    colds = []
    for i in range(a.cold_starts):
        print(f"cold start {i + 1}: waiting for zero replicas (since {now()})")
        t_idle = time.perf_counter()
        while replicas() > 0:               # wait out the scale-down cooldown
            if time.perf_counter() - t_idle > a.max_idle_wait_s:
                raise SystemExit(f"app did not scale to zero within {a.max_idle_wait_s} s")
            time.sleep(30)
        idle_s = time.perf_counter() - t_idle
        window_start = now()                # zero replicas confirmed; nobody else should call now
        t0 = time.perf_counter()
        health = requests.get(f"{url}/health", timeout=600)
        ready_s = time.perf_counter() - t0
        sid = requests.get(f"{url}/samples", timeout=600).json()[0]["id"]
        first, first_rt = predict(data={"sample_id": sid})
        colds.append({"health_status": health.status_code, "zero_to_health_s": ready_s,
                      "first_predict_round_trip_ms": first_rt, "first_predict_server_ms": first["timings_ms"]["total"],
                      "waited_for_zero_s": idle_s, "window_start": window_start, "window_end": now()})
        print(f"cold start {i + 1}: {ready_s:.1f} s to /health, first predict {first_rt:.0f} ms "
              f"(waited {idle_s:.0f} s for zero replicas; window {window_start} -> {colds[-1]['window_end']})")
    cold = {"runs": colds, "zero_to_health_s": stats([c["zero_to_health_s"] for c in colds]) if colds else None,
            "note": "from a confirmed zero-replica state; includes scheduling, image pull if not cached, model load"}

    # ---- warm latency on the gallery ----
    ids = [s["id"] for s in requests.get(f"{url}/samples", timeout=600).json()]
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

    health = requests.get(f"{url}/health", timeout=600).json()
    report = {"service": "Azure Container Apps", "url": url, "region": app["location"],
              "revision": app["properties"]["latestRevisionName"], "image": container["image"],
              "image_compressed_mb": round(compressed_size(container["image"]) / 1e6, 1),
              "hardware": {"vcpu": container["resources"]["cpu"], "memory": container["resources"]["memory"],
                           "scale": app["properties"]["template"]["scale"],
                           "cpu_model": health.get("cpu_model"), "cgroup_cpu_limit": health.get("cpu_limit"),
                           "host_cpu_count": health.get("host_cpu_count"), "ort_threads": health.get("threads")},
              "cold_start": cold, "samples": samples, "uploads_12mp": uploads, "health": health,
              "display": {"processing_time_s": round(samples["server_ms"]["total"]["median"] / 1000, 2),
                          "upload_processing_time_s": round(uploads["server_total_ms"]["median"] / 1000, 2),
                          "cold_start_s": None if not colds else round(cold["zero_to_health_s"]["median"])}}
    os.makedirs(a.report_dir, exist_ok=True)
    path = os.path.join(a.report_dir, "space_check.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"hardware {report['hardware']}\nimage {report['image_compressed_mb']} MB compressed -> {path}")


if __name__ == "__main__":
    main()
