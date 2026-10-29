import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from depth_model import lead_normalisation, to_input_tensor
from pipeline import fuse_labels      # re-exported: defined once, in the torch-free pipeline

SEG_CKPT = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


class Segmenter:
    def __init__(self, checkpoint=SEG_CKPT, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SegformerForSemanticSegmentation.from_pretrained(checkpoint)
        self.model.to(self.device).eval()
        self.mean, self.std = lead_normalisation(checkpoint)
        cfg = self.model.config
        self.names = [cfg.id2label[i] for i in range(cfg.num_labels)]   # class id -> name

    def ids(self, class_names):
        """Class names (e.g. ["road", "sidewalk"]) -> their integer ids."""
        return [self.names.index(c) for c in class_names]

    @torch.no_grad()
    def predict(self, image_rgb):
        """
        uint8 (H, W, 3) RGB -> (H, W) class ids.

        SegFormer outputs one score per class at 1/4 resolution; the scores
        are upsampled to full size first and the best class taken after, so
        boundaries stay smooth. The image is NOT resized beforehand: any size
        works, and keeping the pixel grid means each label lines up exactly
        with the depth map's pixel.
        """
        x = to_input_tensor(image_rgb, self.mean, self.std).unsqueeze(0).to(self.device)
        logits = self.model(pixel_values=x).logits                      # (1, C, H/4, W/4)
        logits = F.interpolate(logits, size=image_rgb.shape[:2], mode="bilinear",
                               align_corners=False)
        return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)