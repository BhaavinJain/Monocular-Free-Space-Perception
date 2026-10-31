"""
Publish the public Hugging Face model repo: ONNX models + JSONs + fine-tuned PyTorch
weights + gallery + model card.

    python -m deploy.deploy

Authentication is whatever `hf auth login` stored in the user's cache; this
script never reads, prints or writes the token itself.

It prints the new commit sha: put it in deploy/Dockerfile (MODEL_REVISION) and
commit, so the server image always builds against one reviewable model version.
(It also had a --space step for a Hugging Face Docker Space; removed, since
Docker Spaces need PRO and the server runs on Azure Container Apps.)
"""
import argparse
import os

from huggingface_hub import CommitOperationAdd, HfApi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx_dir", default="checkpoints/onnx")
    p.add_argument("--samples", default="backend/samples")
    p.add_argument("--weights", default="checkpoints/depth_best.pt",
                   help="fine-tuned PyTorch state_dict, published so the full-width results reproduce")
    a = p.parse_args()
    api = HfApi()
    user = api.whoami()["name"]
    models_repo = f"{user}/monocular-freespace-models"

    ops = [CommitOperationAdd(f, os.path.join(a.onnx_dir, f))
           for f in ("depth.onnx", "depth.json", "segformer.onnx", "segformer.json")]
    ops.append(CommitOperationAdd("depth_best.pt", a.weights))
    for root, _, files in os.walk(a.samples):
        for f in files:
            local = os.path.join(root, f)
            ops.append(CommitOperationAdd("samples/" + os.path.relpath(local, a.samples).replace(os.sep, "/"), local))
    with open("deploy/MODEL_CARD.md", encoding="utf-8") as fh:
        card = fh.read().replace("HF_USER", user)
    ops.append(CommitOperationAdd("README.md", card.encode()))
    api.create_repo(models_repo, repo_type="model", private=False, exist_ok=True)
    info = api.create_commit(models_repo, ops, commit_message="ONNX models, KITTI gallery, model card")
    print(f"https://huggingface.co/{models_repo}  ({len(ops)} files)\ncommit sha: {info.oid}")


if __name__ == "__main__":
    main()
