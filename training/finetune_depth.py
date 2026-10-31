import csv
import os
import random
import time

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForDepthEstimation

from models.depth_model import METRIC_CKPT, to_input_tensor, lead_normalisation
from freespace.metrics import compute_depth_metrics, average_metrics

MIN_D, MAX_D = 1e-3, 80.0
EVAL_CROP = (350, 1218)          # (H, W), multiples of 14, fits every KITTI date

DATA_ROOT = "data"                   
DEPTH_ROOT = "data/kitti_depth"       
CHECKPOINT = METRIC_CKPT              
EPOCHS = 5                          
BATCH_SIZE = 4                        
LR_ENCODER, LR_DECODER = 5e-6, 5e-5  
FREEZE_ENCODER = False               
TRAIN_STEP = 1                       
VAL_STEP = 10                         
WORKERS = 2                           
OUT_DIR = "checkpoints"              
SEED = 0

class KittiDepthPairs(Dataset):
    def __init__(self, root, depth_root, split, train, mean, std,
                 crop=(350, 700), frame_step=1):
        self.train = train
        self.mean, self.std = mean, std

        self.aug = A.Compose([
            A.RandomCrop(height=crop[0], width=crop[1]),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        ])
        self.items, skipped = [], []
        split_dir = os.path.join(depth_root, split)
        for name in sorted(os.listdir(split_dir)):
            img_dir = os.path.join(root, name[:10], name, "image_02", "data")
            if not os.path.isdir(img_dir):
                skipped.append(name)
                continue
            gt_dir = os.path.join(split_dir, name, "proj_depth", "groundtruth", "image_02")
            for fn in sorted(os.listdir(gt_dir))[::frame_step]:
                img = os.path.join(img_dir, fn)
                if os.path.exists(img):
                    self.items.append((img, os.path.join(gt_dir, fn)))
        print(f"[{split}] {len(self.items)} pairs from "
              f"{len(set(os.path.dirname(os.path.dirname(i[0])) for i in self.items))} drives; "
              f"{len(skipped)} drives skipped (raw images not downloaded)")
        if not self.items:
            raise SystemExit(f"No {split} pairs: download raw drives that appear in "
                             f"{split_dir} and place them under DATA_ROOT.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_path, gt_path = self.items[i]
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        H, W = gt.shape

        if self.train:
            out = self.aug(image=img, mask=gt)
            img, gt = out["image"], out["mask"]
        else:
            ch, cw = EVAL_CROP
            y0, x0 = H - ch, (W - cw) // 2
            img, gt = img[y0:y0 + ch, x0:x0 + cw], gt[y0:y0 + ch, x0:x0 + cw]

        x = to_input_tensor(np.ascontiguousarray(img), self.mean, self.std)
        y = torch.from_numpy(np.ascontiguousarray(gt)).unsqueeze(0)
        return x, y


def silog_loss(pred, gt, lam=0.85):
    """Masked scale-invariant log loss. pred, gt: (B, H, W) float32."""
    mask = (gt > MIN_D) & (gt < MAX_D)
    g = torch.log(pred[mask].clamp(min=MIN_D)) - torch.log(gt[mask])
    # x10 is a conventional rescaling so the loss is O(1), not O(0.1)
    return 10.0 * torch.sqrt((g ** 2).mean() - lam * g.mean() ** 2 + 1e-8)


def forward(model, x, size):
    """Run the network and make sure the output matches the target size."""
    out = model(pixel_values=x).predicted_depth            # (B, h, w)
    if out.shape[-2:] != size:
        out = F.interpolate(out.unsqueeze(1), size=size, mode="bilinear",
                            align_corners=False).squeeze(1)
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    per = []
    for x, y in loader:
        pred = forward(model, x.to(device), y.shape[-2:]).float().cpu().numpy()
        for p, g in zip(pred, y[:, 0].numpy()):
            per.append(compute_depth_metrics(p, g, MIN_D, MAX_D))
    model.train()
    return average_metrics(per)


def main():

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if use_amp else
                                 "  — WARNING: training on CPU will be extremely slow"))

    mean, std = lead_normalisation(CHECKPOINT)
    train_ds = KittiDepthPairs(DATA_ROOT, DEPTH_ROOT, "train", True, mean, std,
                               frame_step=TRAIN_STEP)
    val_ds = KittiDepthPairs(DATA_ROOT, DEPTH_ROOT, "val", False, mean, std,
                             frame_step=VAL_STEP)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=WORKERS,
                          pin_memory=use_amp, drop_last=True)
    val_dl = DataLoader(val_ds, 1, shuffle=False, num_workers=WORKERS)

    model = AutoModelForDepthEstimation.from_pretrained(CHECKPOINT).to(device)

    # Two learning rates: the pretrained encoder already knows how to see,
    # so it moves 10x more slowly than the decoder, which adapts to KITTI.
    enc = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    dec = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    if FREEZE_ENCODER:
        for prm in enc:
            prm.requires_grad = False
        groups = [{"params": dec, "lr": LR_DECODER}]
    else:
        groups = [{"params": enc, "lr": LR_ENCODER},
                  {"params": dec, "lr": LR_DECODER}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)

    total = EPOCHS * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: (1 - step / total) ** 0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, "finetune_log.csv")
    log_file = open(log_path, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["epoch", "train_loss", "val_abs_rel", "val_rmse", "val_d1", "seconds"])

    base = evaluate(model, val_dl, device)
    print(f"before fine-tuning: val abs_rel {base['abs_rel']:.4f}  "
          f"rmse {base['rmse']:.3f}  d1 {base['d1']:.4f}")
    best = base["abs_rel"]

    model.train()
    for epoch in range(1, EPOCHS + 1):
        t0, losses = time.time(), []
        for it, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)[:, 0]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                pred = forward(model, x, y.shape[-2:])
            loss = silog_loss(pred.float(), y)        # loss in float32: logs are touchy

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())
            if it % 50 == 0:
                print(f"  epoch {epoch} iter {it}/{len(train_dl)} loss {loss.item():.4f}")

        val = evaluate(model, val_dl, device)
        secs = time.time() - t0
        log.writerow([epoch, np.mean(losses), val["abs_rel"], val["rmse"], val["d1"], round(secs)])
        log_file.flush()                  # written to disk now, even if a later epoch crashes
        print(f"epoch {epoch}: train loss {np.mean(losses):.4f} | val abs_rel {val['abs_rel']:.4f} "
              f"rmse {val['rmse']:.3f} d1 {val['d1']:.4f} | {secs:.0f}s")
        if val["abs_rel"] < best:
            best = val["abs_rel"]
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "depth_best.pt"))
            print(f"  new best -> {OUT_DIR}/depth_best.pt")

    log_file.close()
    print(f"\nval abs_rel: {base['abs_rel']:.4f} before -> {best:.4f} best after")
    print(f"log -> {log_path}")


if __name__ == "__main__":
    main()