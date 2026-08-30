# Katsu-Hybrid-AIGC-Detector
Katsu is a lightweight, self-supervised-backbone hybrid model for detecting AI-generated images, built specifically to stay accurate after the image has been through real-world redistribution (recompression, resizing, cropping, noise). It combines a **frozen DINOv2-S/14 backbone** with two complementary, lightweight artifact-detection branches: an **NPR (upsampling-residual) branch** and a **texture-statistics (LBP/GLCM) branch**, fused through an **importance-weighted gating head**. Only **33,842 trainable parameters** sit on top of the frozen backbone. This model is built for Tik Tok's annual hackathon **TechJam 2026**.

<img width="1024" height="1059" alt="Katsu_Summary_Visual" src="https://github.com/user-attachments/assets/6d2cc0fd-172f-41be-a4e6-af7d3777ba73" />


## How This Solution Addresses the Problem Statement

Platforms like TikTok already auto-label AI-generated content via C2PA Content Credentials and invisible watermarks, but that provenance metadata is stripped the moment content is screenshotted, re-encoded, or reposted off-platform. Once that happens, there's no metadata left to check.

Katsu is a **content-based fallback layer** for exactly that scenario: it doesn't rely on any embedded signal, only the pixels themselves, and it's trained specifically to keep working after the kinds of transformations that strip metadata in the first place. That's the "redistribution robustness" this problem statement is testing, not just detecting AI-generated images in their original, clean form.

## Installation and Setup

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

The first inference run downloads the pretrained DINOv2 ViT-S/14 backbone through PyTorch Hub, so internet access is required unless the backbone has already been cached.

## Required inference interface

The submission requirement is a script that accepts an image directory and writes one confidence score per image to JSON.

```bash
python predict.py ./path/to/images \
    --checkpoint ./hybrid_checkpoint.pt \
    --output ./predictions.json
```

The image directory is searched recursively. Supported formats are JPG, JPEG, PNG, BMP, and WEBP.

### JSON output

The output contains `image_path` and `pred` for every successfully processed image:

```json
[
  {
    "image_path": "./path/to/images/example1.jpg",
    "pred": 0.9234
  },
  {
    "image_path": "./path/to/images/example2.jpg",
    "pred": 0.0871
  }
]
```

`pred` is the model's sigmoid output and is interpreted as the estimated likelihood of the image being AIGC-generated. Higher values indicate stronger model confidence that the image is generated.


## Notebook

`KatsuDemo.ipynb` contains the original experimental pipeline, including dataset preparation, training, evaluation/robustness analysis, error analysis, and the interactive demonstration.

For reproducible command-line inference, use `predict.py` rather than executing the notebook.

## Model architecture

The inference pipeline is:

```text
                    ┌── DINOv2 ViT-S/14 ───────────┐
Input image ────────┼── NPR residual encoder ──────┼── Importance-weighted fusion ── sigmoid ── pred
                    └── LBP + GLCM texture ────────┘
```

The DINOv2 backbone is frozen. The trained components are the residual encoder, texture projection, and fusion head.


