""" Run Depth Anything V2 on an RGB image
    Two kinds of checkpoint : 
        RELATIVE_CKPT  "Depth-Anything-V2-Small-hf" 
            Outputs inverse depth (bigger = closer) with an unknown scale and shift per image
            One constraint (camera height) cannot fix two unknowns, so the
            plain s = h/d method does NOT apply to this output
        
        METRIC_CKPT  "Depth-Anything-V2-Metric-Outdoor-Small-hf"
            Outputs metric depth (bigger = farther) with a known scale and shift per image
        
    This project uses METRIC_CKPT by default
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation      # Give it a model name, it downloads that model's config file, reads which architecture it is, and builds the right one

RELATIVE_CKPT = "depth-anything/Depth-Anything-V2-Small-hf"
METRIC_CKPT = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
PATCH = 14

def lead_normalisation(checkpoint):
    """Read the per-channel mean and std the checkpoint was trained with, from
    the preprocessing config that ships with it. Inputs must be normalised
    exactly this way or every feature the network computes is subtly off."""

    proc = AutoImageProcessor.from_pretrained(checkpoint)
    mean = torch.tensor(proc.image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(proc.image_std, dtype=torch.float32).view(3, 1, 1)
    return mean, std

def to_input_tensor(image_rgb, mean, std):
    """Convert a uint8 RGB image to a normalised float tensor. Pytorch expects channels first """
    x = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0            # reorders the axes from (H, W, 3) to (3, H, W)
    return (x - mean) / std

class DepthModel:
    def __init__(self, checkpoint = METRIC_CKPT, weights = None, device = None):
        # checkpoint : Hugging Face model name (architecture + pretrained weights)
        # weights    : optional .pt file from finetune_depth.py, loaded on top
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForDepthEstimation.from_pretrained(checkpoint)
        self.mean, self.std = lead_normalisation(checkpoint)
        if weights is not None:
            state = torch.load(weights, map_location="cpu")
            self.model.load_state_dict(state)                            # dictionary mapping each layer's name to its weight tensor
        self.model.to(self.device).eval()

        # "metric" or "relative", read from the checkpoint's own config
        self.output_type = getattr(self.model.config,
                                   "depth_estimation_type", "relative")

        @torch.no_grad()
        def predict(self, image_rgb):
            # uint8 (H, W, 3) RGB -> float32 (H, W) prediction, same size
            H, W = image_rgb.shape[:2]
            h14 = max(PATCH, int(round(H / PATCH)) * PATCH)
            w14 = max(PATCH, int(round(W / PATCH)) * PATCH)

            x = to_input_tensor(image_rgb, self.mean, self.std).unsqueeze(0)    # (1, 3, H, W)
            if (h14, w14) != (H, W):
                x = F.interpolate(x, size=(h14, w14), mode="bilinear" ,align_corners=False)
            x = x.to(self.device)
            out = self.model(pixel_values=x).predicted_depth      # (1, h14, w14)
            out = F.interpolate(out.unsqueeze(1), size=(H, W), mode="bilinear",
                            align_corners=False)[0, 0]
            return out.float().cpu().numpy()                # Copying from GPU to CPU memory, and converting from torch tensor to numpy array