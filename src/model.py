"""Model and feature extraction code for the Katsu AIGC detector.

The implementation mirrors the model used in KatsuDemo.ipynb:
- frozen DINOv2 ViT-S/14 visual features
- upsampling-residual (NPR) artifact features
- LBP/GLCM texture statistics
- importance-weighted feature fusion
"""

import io
from typing import Dict, List

import numpy as np
import skimage.feature as skf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINO_EMBED_DIM = 384

DINO_PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ResidualEncoder(nn.Module):
    """Small CNN that encodes the image upsampling residual."""

    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ImportanceWeightedFusion(nn.Module):
    """Project each branch into a common space and learn branch weights."""

    def __init__(self, branch_dims, common_dim: int = 64):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(d, common_dim) for d in branch_dims]
        )
        self.gate = nn.Linear(common_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(common_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, branch_feats):
        projected = [proj(f) for proj, f in zip(self.projections, branch_feats)]
        stacked = torch.stack(projected, dim=1)
        weights = F.softmax(self.gate(stacked).squeeze(-1), dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(fused).squeeze(-1), weights


def images_to_raw_batch(pil_imgs: List[Image.Image], size: int = 224):
    """Resize RGB PIL images and convert them to [B, C, H, W] tensors."""
    arrs = [
        np.array(img.resize((size, size))).astype(np.float32) / 255.0
        for img in pil_imgs
    ]
    return torch.tensor(np.stack(arrs)).permute(0, 3, 1, 2).to(DEVICE)


def compute_npr_residual_batch(raw_batch, scale: float = 0.5):
    """Compute a simple downsample-then-upsample residual."""
    down = F.interpolate(
        raw_batch, scale_factor=scale, mode="bilinear", align_corners=False
    )
    up = F.interpolate(
        down, size=raw_batch.shape[-2:], mode="bilinear", align_corners=False
    )
    return raw_batch - up


def extract_texture_features(pil_img: Image.Image):
    """Return the 26-dimensional LBP + GLCM texture feature vector."""
    gray = np.array(pil_img.convert("L").resize((128, 128)))
    lbp = skf.local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)

    glcm = skf.graycomatrix(
        gray,
        distances=[1, 3],
        angles=[0, np.pi / 2],
        symmetric=True,
        normed=True,
    )
    glcm_feats = np.concatenate(
        [skf.graycoprops(glcm, p).flatten()
         for p in ["contrast", "homogeneity", "energy", "correlation"]]
    )
    return np.concatenate([lbp_hist, glcm_feats]).astype(np.float32)


def build_models():
    """Construct the frozen DINOv2 backbone and trainable detector heads."""
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEVICE)
    dinov2.eval()
    for p in dinov2.parameters():
        p.requires_grad = False

    residual_encoder = ResidualEncoder(out_dim=32).to(DEVICE)
    texture_proj = nn.Linear(26, 32).to(DEVICE)
    fusion_head = ImportanceWeightedFusion(
        branch_dims=[DINO_EMBED_DIM, 32, 32], common_dim=64
    ).to(DEVICE)

    return dinov2, residual_encoder, texture_proj, fusion_head


def load_models(checkpoint_path: str) -> Dict[str, nn.Module]:
    """Load the detector and trained checkpoint into evaluation mode."""
    dinov2, residual_encoder, texture_proj, fusion_head = build_models()

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    residual_encoder.load_state_dict(checkpoint["residual_encoder_state"])
    texture_proj.load_state_dict(checkpoint["texture_proj_state"])
    fusion_head.load_state_dict(checkpoint["fusion_head_state"])

    residual_encoder.eval()
    texture_proj.eval()
    fusion_head.eval()

    return {
        "dinov2": dinov2,
        "residual_encoder": residual_encoder,
        "texture_proj": texture_proj,
        "fusion_head": fusion_head,
    }


def predict_image(pil_img: Image.Image, models: Dict[str, nn.Module]) -> float:
    """Return the probability that one image is AIGC-generated."""
    pil_img = pil_img.convert("RGB")

    with torch.no_grad():
        dino_batch = DINO_PREPROCESS(pil_img).unsqueeze(0).to(DEVICE)
        dino_feats = models["dinov2"](dino_batch)

        raw_batch = images_to_raw_batch([pil_img])
        residuals = compute_npr_residual_batch(raw_batch)
        npr_feats = models["residual_encoder"](residuals)

        texture_vec = torch.tensor(
            extract_texture_features(pil_img), device=DEVICE
        )
        texture_feats = models["texture_proj"](texture_vec).unsqueeze(0)

        logits, _ = models["fusion_head"](
            [dino_feats, npr_feats, texture_feats]
        )
        return float(torch.sigmoid(logits).item())
