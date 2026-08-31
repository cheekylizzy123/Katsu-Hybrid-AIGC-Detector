# Katsu-Hybrid-AIGC-Detector
Katsu is a lightweight hybrid model for detecting **AI-generated images (AIGC)** using visual evidence contained directly in the image pixels. **Link to Demo**: https://katsu-demogit-ix9aua9mfr9fv4akep7dku.streamlit.app/

The model is designed specifically for **real-world redistribution**, where images may be recompressed, resized, cropped, or affected by noise. Rather than relying on metadata, watermarks, or provenance signals, Katsu combines semantic visual representations with complementary image-artifact and texture features. **Robustness Evaluation Summary** and **Error Analysis Note** are both included as **Robustness Evaluation.md**.

The system uses a **frozen DINOv2 ViT-S/14 backbone** together with two lightweight artifact-detection branches:

* **NPR residual branch** — captures image reconstruction and resampling artifacts.
* **Texture-statistics branch** — extracts LBP and GLCM texture features.
* **Importance-weighted fusion head** — combines the complementary feature representations.

Only **33,842 trainable parameters** are used on top of the frozen backbone.

This project was developed for **TikTok TechJam 2026**. 
<br><br>
## How This Solution Addresses the Problem Statement

Platforms like TikTok already auto-label AI-generated content via C2PA Content Credentials and invisible watermarks, but that provenance metadata is stripped the moment content is screenshotted, re-encoded, or reposted off-platform. Once that happens, there's no metadata left to check.

Katsu is a **content-based fallback layer** for exactly that scenario: it doesn't rely on any embedded signal, only the pixels themselves, and it's trained specifically to keep working after the kinds of transformations that strip metadata in the first place. That's the "redistribution robustness" this problem statement is testing, not just detecting AI-generated images in their original, clean form.
<br><br>
## Repository Structure

```
katsu-hybrid-aigc-detector/
├── README.md
├── KatsuTraining.ipynb        # Full experimental/training/analysis notebook
├── predict.py                 # Directory → JSON inference script
├── hybrid_checkpoint.pt       # Trained model checkpoint
├── requirements.txt           # Dependencies for predict.py
├── requirements-training.txt  # Dependencies for KatsuTraining.ipynb
│
├── src/
│   ├── __init__.py
│   └── model.py                # Model architecture and feature extraction
│
└── katsu-streamlit-demo/       # Demo files
    ├── README.md
    ├── app.py
    ├── hybrid_checkpoint.pt
    ├── predict.py
    └── requirements.txt
```
<br><br>
## Installation and Setup: Inference Interface

### Option A: Google Colab (Recommended)

**Link to Colab:** https://colab.research.google.com/drive/1RCrPGQaUM2nOr0uymuZAkN259atOSKDj?usp=sharing

**1. Install dependencies in Colab notebook:** 

```python
!pip install -q scikit-image opencv-python-headless
```

**2. Upload `predict.py` and `hybrid_checkpoint.pt`:**

```python
from google.colab import files
uploaded = files.upload()
```

**3. Upload your test images (.zip file):** 

```python
from google.colab import files
uploaded_zip = files.upload() 
```
```python
import zipfile
with zipfile.ZipFile('images.zip', 'r') as z: 
    z.extractall('/content/test_images')
# change 'image.zip' to 'file_name.zip'
```

**4. Run the scoring script:**

```python
!python predict.py /content/test_images --checkpoint hybrid_checkpoint.pt --output predictions.json
```

**5. Download the results:**

```python
from google.colab import files
files.download('predictions.json')
```

### Option B: Python Environment

**1. Clone the repository:**

```bash
git clone https://github.com/cheekylizzy123/Katsu-Hybrid-AIGC-Detector.git
cd Katsu-Hybrid-AIGC-Detector
```

**2. Install dependencies** (Python 3.10+ recommended):

```bash
pip install -r requirements.txt
```

**3. Run the script** on any folder of images:

```bash
python predict.py /path/to/image_folder --checkpoint hybrid_checkpoint.pt --output predictions.json
```
Accepts `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp` files. Unreadable files are skipped with a warning rather
than stopping the run.

### Results: Output Format

```json
[
  { "image_path": "test_images/photo1.jpg", "pred": 0.0213 },
  { "image_path": "test_images/photo2.png", "pred": 0.9142 }
]
```

The `pred` value ranges from **0 to 1**. Higher values indicate stronger model confidence that the image is AI-generated.
<br><br>
## Installation and Setup: Reproduce Training & Evaluation

**Link to Colab:** https://colab.research.google.com/drive/1e-8D6em4XKlE877zceaQa7YKcWTEHQG2?usp=sharing 

**1. Install dependencies:** 

```bash
pip install -r requirements-training.txt
# default 
```

**2. Open `KatsuTraining.ipynb`** in **Colab** and run the cells in order

#### Self-Transformed Dataset

The dataset was transformed from 5k raw real images from COCO train2017 and 6k raw AI-generated images, split evenly across SD1.5, Midjourney, and ADM, to form 33k transformed images. The inclusion of architecturally distinct generators (latent diffusion, closed-source/black-box, and pixel-space diffusion) forces the detector to learn generalisable generation artifacts rather than memorising the fingerprint of a single generator.

**Link to Self-Transformed Dataset:** https://www.kaggle.com/datasets/shxrlenee/aigc-detection-dataset
<br><br>
## Model Architecture

Katsu combines three complementary feature branches:

```text
                         ┌──────────────────────────┐
                         │   DINOv2 ViT-S/14        │
                         │   Semantic Features      │
                         └────────────┬─────────────┘
                                      │
Input Image ──────────────────────────┼─────────────────────┐
                                      │                     │
                         ┌────────────▼─────────────┐       │
                         │   NPR Residual Branch    │       │
                         │   Image Artifacts        │       │
                         └────────────┬─────────────┘       │
                                      │                     │
                         ┌────────────▼─────────────┐       │
                         │   LBP + GLCM Branch      │       │
                         │   Texture Statistics     │       │
                         └────────────┬─────────────┘       │
                                      │                     │
                                      └──────────┬──────────┘
                                                 │
                                  ┌──────────────▼──────────────┐
                                  │ Importance-Weighted Fusion  │
                                  │            Head             │
                                  └──────────────┬──────────────┘
                                                 │
                                              Sigmoid
                                                 │
                                                 ▼
                                          AIGC Confidence
```

### 1. DINOv2 Branch

A pretrained **DINOv2 ViT-S/14** backbone provides high-level visual representations.

The backbone is frozen during training, allowing the detector to build a lightweight classification system on top of the pretrained representation.

### 2. NPR Residual Branch

The residual branch extracts image artifacts associated with reconstruction, interpolation, and resampling.

This branch is intended to capture lower-level signals that may complement the semantic representation produced by DINOv2.

### 3. Texture Branch

The texture branch extracts handcrafted statistical descriptors:

* **Local Binary Patterns (LBP)**
* **Gray-Level Co-occurrence Matrix (GLCM)**

These features provide additional information about local texture and spatial relationships between pixel intensities.

### 4. Importance-Weighted Fusion

The outputs of the three branches are projected into a common representation and combined through an importance-weighted fusion head.

The final representation is passed through a sigmoid function to produce the AIGC confidence score.
<br><br>
## Limitations & Potential Improvements

### 1. Frozen DINOv2 backbone caps the accuracy
Keeping DINOv2 fully frozen and total trainable params at 33.8K was a deliberate efficiency choice, but it also means the model can only ever re-weight existing DINOv2/NPR/texture features — it can't learn new representations, which likely explains the accuracy plateau in the 81–87% range across every evaluation condition

**Improvement: Selectively unfreezing DINOv2's last block (small LR) as an ablation & compare evaluation**

### 2. Lack of language-aligned semantic representations
Originally scoped as a dual-backbone design (CLIP ViT-L/14 + DINOv2 ViT-S/14), the architecture was reduced to DINOv2 only due to VRAM limits, which may reduce the model's ability to catch semantically implausible generations

**Improvement: Reintroduce CLIP ViT-L/14 as a second backbone given access to more VRAM, or explore mitigations such as gradient checkpointing, mixed precision, or sequential backbone forward passes**
<br><br>
## Team Member Contributions

- **Sharlene** — Data pipeline (WildFake/CIFAKE/SID_Set/Self-Transformed preprocessing, CLIP feature caching)
- **Elizabeth** — Model architecture (DINOv2 backbone, NPR residual branch, texture feature extraction, fusion head)
- **Kai En** — Training loop, hyperparameter tuning, checkpoint management, manifest building
- **Jing Jing** — Evaluation (robustness testing, error analysis, OOD validation set), README and documentation

