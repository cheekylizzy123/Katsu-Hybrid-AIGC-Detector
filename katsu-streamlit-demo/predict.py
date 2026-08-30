import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import skimage.feature as skf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINO_EMBED_DIM = 384

dino_preprocess = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

def find_images(directory):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(paths)

class ResidualEncoder(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class ImportanceWeightedFusion(nn.Module):
    def __init__(self, branch_dims, common_dim=64):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(d, common_dim) for d in branch_dims])
        self.gate = nn.Linear(common_dim, 1)
        self.classifier = nn.Sequential(nn.Linear(common_dim, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, branch_feats):
        projected = [proj(f) for proj, f in zip(self.projections, branch_feats)]
        stacked = torch.stack(projected, dim=1)
        weights = F.softmax(self.gate(stacked).squeeze(-1), dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(fused).squeeze(-1), weights

def images_to_raw_batch(pil_imgs, size=224):
    arrs = [np.array(img.resize((size, size))).astype(np.float32) / 255.0 for img in pil_imgs]
    return torch.tensor(np.stack(arrs)).permute(0, 3, 1, 2).to(DEVICE)

def compute_npr_residual_batch(raw_batch, scale=0.5):
    down = F.interpolate(raw_batch, scale_factor=scale, mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=raw_batch.shape[-2:], mode="bilinear", align_corners=False)
    return raw_batch - up

def extract_texture_features(pil_img):
    gray = np.array(pil_img.convert("L").resize((128, 128)))
    lbp = skf.local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    glcm = skf.graycomatrix(gray, distances=[1, 3], angles=[0, np.pi / 2], symmetric=True, normed=True)
    glcm_feats = np.concatenate([skf.graycoprops(glcm, p).flatten()
                                  for p in ["contrast", "homogeneity", "energy", "correlation"]])
    return np.concatenate([lbp_hist, glcm_feats]).astype(np.float32)

def load_models(checkpoint_path):
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE)
    dinov2.eval()
    for p in dinov2.parameters():
        p.requires_grad = False

    residual_encoder = ResidualEncoder(out_dim=32).to(DEVICE)
    texture_proj = nn.Linear(26, 32).to(DEVICE)
    fusion_head = ImportanceWeightedFusion(branch_dims=[DINO_EMBED_DIM, 32, 32], common_dim=64).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    residual_encoder.load_state_dict(ckpt["residual_encoder_state"])
    texture_proj.load_state_dict(ckpt["texture_proj_state"])
    fusion_head.load_state_dict(ckpt["fusion_head_state"])
    residual_encoder.eval(); texture_proj.eval(); fusion_head.eval()

    return {"dinov2": dinov2, "residual_encoder": residual_encoder,
            "texture_proj": texture_proj, "fusion_head": fusion_head}

def predict_image(pil_img, models):
    pil_img = pil_img.convert("RGB")
    dino_batch = dino_preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        dino_feats = models["dinov2"](dino_batch)

        raw_batch = images_to_raw_batch([pil_img])
        residuals = compute_npr_residual_batch(raw_batch)
        npr_feats = models["residual_encoder"](residuals)

        tex_vec = torch.tensor(extract_texture_features(pil_img)).to(DEVICE)
        tex_feats = models["texture_proj"](tex_vec).unsqueeze(0)

        logits, _ = models["fusion_head"]([dino_feats, npr_feats, tex_feats])
        prob_fake = torch.sigmoid(logits).item()

    return prob_fake
