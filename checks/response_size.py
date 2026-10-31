"""
Size of the /predict response, raw JSON and as sent (gzip), per request.

    python -m checks.response_size --tag before     (server running on --url)

Why: the frontend downloads the whole JSON before it can draw anything, so the
response size is part of the browser time. Run once per server version with a
different --tag; results are kept side by side in report/response_size.json.
Inputs: the 6 gallery samples, and the full KITTI frame from
backend/test_images uploaded with h = 1.65, hfov = 80 deg (an upload that is
not refused, so all metric fields are present).
"""
import argparse
import gzip
import json
import os

import numpy as np
import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:7860")
    p.add_argument("--tag", required=True)
    p.add_argument("--upload", default="backend/test_images/kitti_0023_frame5.png")
    p.add_argument("--out", default="report/response_size.json")
    a = p.parse_args()

    ids = [s["id"] for s in requests.get(f"{a.url}/samples").json()]
    reqs = [(i, {"data": {"sample_id": i}}) for i in ids]
    reqs.append((os.path.basename(a.upload), {"data": {"camera_height": 1.65, "hfov_deg": 80},
                                               "files": {"file": open(a.upload, "rb").read()}}))
    runs = []
    for name, kw in reqs:
        r = requests.post(f"{a.url}/predict", headers={"Accept-Encoding": "gzip"}, stream=True, **kw)
        wire = r.raw.read(decode_content=False)          # bytes as sent over the network
        body = gzip.decompress(wire) if r.headers.get("Content-Encoding") == "gzip" else wire
        res = json.loads(body)
        runs.append({"request": name, "status": res["status"], "json_kb": len(body) / 1024,
                     "gzip_kb": len(wire) / 1024, "has_hover": "hover" in res})
        print(f"{name}: {res['status']}, {len(body) / 1024:.0f} KB JSON, {len(wire) / 1024:.0f} KB sent")

    report = json.load(open(a.out)) if os.path.exists(a.out) else {}
    report[a.tag] = {"url": a.url, "runs": runs,
                     "median_json_kb": float(np.median([r["json_kb"] for r in runs])),
                     "median_gzip_kb": float(np.median([r["gzip_kb"] for r in runs]))}
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"median: {report[a.tag]['median_json_kb']:.0f} KB JSON, {report[a.tag]['median_gzip_kb']:.0f} KB sent -> {a.out}")


if __name__ == "__main__":
    main()
