"""
Publish to Hugging Face, in two separate steps:

    python space/deploy.py --models   # public model repo: ONNX + JSONs + gallery + model card
    python space/deploy.py --space    # the Docker Space: Dockerfile, Space card, server code only

Authentication is whatever `hf auth login` stored in the user's cache; this
script never reads, prints or writes the token itself.

--models prints the new commit sha: put it in space/Dockerfile (MODEL_REVISION)
and commit, so the Space always builds against one reviewable model version.
--space refuses to run while the Dockerfile still holds a placeholder.
"""
import argparse
import os

from huggingface_hub import CommitOperationAdd, HfApi

# exactly what the server imports — nothing for training, no KITTI loader
SPACE_CODE = ["pipeline.py", "bev.py", "camera.py", "ground_plane.py", "planner.py",
              "pointcloud.py", "scale.py", "metrics.py", "backend/app.py", "space/requirements.txt"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", action="store_true")
    p.add_argument("--space", action="store_true")
    p.add_argument("--onnx_dir", default="checkpoints/onnx")
    p.add_argument("--samples", default="backend/samples")
    a = p.parse_args()
    api = HfApi()
    user = api.whoami()["name"]
    models_repo, space_repo = f"{user}/monocular-freespace-models", f"{user}/monocular-freespace"

    if a.models:
        ops = [CommitOperationAdd(f, os.path.join(a.onnx_dir, f))
               for f in ("depth.onnx", "depth.json", "segformer.onnx", "segformer.json")]
        for root, _, files in os.walk(a.samples):
            for f in files:
                local = os.path.join(root, f)
                ops.append(CommitOperationAdd("samples/" + os.path.relpath(local, a.samples).replace(os.sep, "/"), local))
        with open("space/MODEL_CARD.md", encoding="utf-8") as fh:
            card = fh.read().replace("HF_USER", user)
        ops.append(CommitOperationAdd("README.md", card.encode()))
        api.create_repo(models_repo, repo_type="model", private=False, exist_ok=True)
        info = api.create_commit(models_repo, ops, commit_message="ONNX models, KITTI gallery, model card")
        print(f"https://huggingface.co/{models_repo}  ({len(ops)} files)\ncommit sha: {info.oid}")

    if a.space:
        with open("space/Dockerfile", encoding="utf-8") as fh:
            dockerfile = fh.read()
        if "HF_USER" in dockerfile or "SET_AFTER_UPLOAD" in dockerfile:
            raise SystemExit("space/Dockerfile still has a placeholder: set MODEL_REPO / MODEL_REVISION first")
        with open("space/README.md", encoding="utf-8") as fh:
            readme = fh.read().replace("HF_USER", user)
        ops = [CommitOperationAdd("Dockerfile", "space/Dockerfile"), CommitOperationAdd("README.md", readme.encode())]
        ops += [CommitOperationAdd(f, f) for f in SPACE_CODE]
        api.create_repo(space_repo, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
        info = api.create_commit(space_repo, ops, repo_type="space", commit_message="Deploy server")
        print(f"https://huggingface.co/spaces/{space_repo}  commit sha: {info.oid}\n"
              f"API: https://{user.lower()}-monocular-freespace.hf.space")


if __name__ == "__main__":
    main()
