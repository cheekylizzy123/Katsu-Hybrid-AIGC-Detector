# Katsu-Hybrid-AIGC-Detector
Katsu is a lightweight hybrid model for detecting **AI-generated images (AIGC)** using visual evidence contained directly in the image pixels.

The model is designed specifically for **real-world redistribution**, where images may be recompressed, resized, cropped, or affected by noise. Rather than relying on metadata, watermarks, or provenance signals, Katsu combines semantic visual representations with complementary image-artifact and texture features.

The system uses a **frozen DINOv2 ViT-S/14 backbone** together with two lightweight artifact-detection branches:

* **NPR residual branch** — captures image reconstruction and resampling artifacts.
* **Texture-statistics branch** — extracts LBP and GLCM texture features.
* **Importance-weighted fusion head** — combines the complementary feature representations.

Only **33,842 trainable parameters** are used on top of the frozen backbone.

This project was developed for **TikTok TechJam 2026**.

<img width="1024" height="1059" alt="Katsu_Summary_Visual" src="https://github.com/user-attachments/assets/6d2cc0fd-172f-41be-a4e6-af7d3777ba73" />


## How This Solution Addresses the Problem Statement

Platforms like TikTok already auto-label AI-generated content via C2PA Content Credentials and invisible watermarks, but that provenance metadata is stripped the moment content is screenshotted, re-encoded, or reposted off-platform. Once that happens, there's no metadata left to check.

Katsu is a **content-based fallback layer** for exactly that scenario: it doesn't rely on any embedded signal, only the pixels themselves, and it's trained specifically to keep working after the kinds of transformations that strip metadata in the first place. That's the "redistribution robustness" this problem statement is testing, not just detecting AI-generated images in their original, clean form.


## Key Features

* 🧠 **DINOv2 visual representation**

  * Frozen DINOv2 ViT-S/14 backbone
  * Captures high-level semantic and visual representations

* 🔍 **NPR residual analysis**

  * Uses an upsampling-based residual representation
  * Captures reconstruction and resampling artifacts

* 📊 **Texture analysis**

  * Local Binary Pattern (LBP) features
  * Gray-Level Co-occurrence Matrix (GLCM) statistics

* ⚖️ **Importance-weighted feature fusion**

  * Combines semantic, residual, and texture information
  * Learns the relative importance of the different branches

* 🛡️ **Redistribution robustness**

  * Designed with real-world image transformations in mind
  * Targets detection after compression, resizing, cropping, and noise

* ⚡ **Lightweight trainable head**

  * Frozen DINOv2 backbone
  * Only 33,842 trainable parameters

* 📁 **Batch directory inference**

  * Accepts an image directory as input
  * Recursively processes supported image formats

* 📄 **JSON predictions**

  * Produces one confidence score per image
  * Output contains `image_path` and `pred`

---

## 📁 Project Structure

```text
katsu-hybrid-aigc-detector/
├── 📄 README.md
├── 📓 KatsuDemo.ipynb          # Full experimental/training/demo notebook
├── 🐍 predict.py               # Directory → JSON inference script
├── 📋 requirements.txt         # Python dependencies
├── 📦 hybrid_checkpoint.pt     # Trained model checkpoint
│
└── 📁 src/
    ├── 📄 __init__.py
    └── 🧠 model.py             # Model architecture and feature extraction
```

### File Descriptions

| File                   | Description                                                                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `KatsuDemo.ipynb`      | Complete experimental pipeline including dataset preparation, training, evaluation, robustness analysis, error analysis, and demonstration |
| `predict.py`           | Standalone inference script required for evaluating a directory of images                                                                  |
| `src/model.py`         | Model architecture and feature-extraction components                                                                                       |
| `hybrid_checkpoint.pt` | Trained weights for the residual, texture, and fusion components                                                                           |
| `requirements.txt`     | Python dependencies required to run the project                                                                                            |
| `README.md`            | Project documentation and usage instructions                                                                                               |

---

# 🚀 Quick Start

## Installation

Python **3.10+** is recommended.

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/katsu-hybrid-aigc-detector.git
cd katsu-hybrid-aigc-detector
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The first inference run downloads the pretrained **DINOv2 ViT-S/14** backbone through PyTorch Hub, so internet access is required unless the model has already been cached locally.

---

# 🔎 Running Inference

Katsu provides a standalone inference script that accepts an **image directory** and produces a JSON file containing a confidence score for each image.

```bash
python predict.py ./path/to/images \
    --checkpoint ./hybrid_checkpoint.pt \
    --output ./predictions.json
```

### Supported Image Formats

The input directory is searched recursively for:

```text
.jpg
.jpeg
.png
.bmp
.webp
```

For example:

```text
images/
├── real/
│   ├── image1.jpg
│   └── image2.png
└── generated/
    ├── image3.jpg
    └── image4.webp
```

Running:

```bash
python predict.py ./images \
    --checkpoint ./hybrid_checkpoint.pt \
    --output predictions.json
```

will process all supported images recursively.

---

# 📄 JSON Output

The inference script produces one prediction for every successfully processed image.

Example:

```json
[
  {
    "image_path": "./images/real/image1.jpg",
    "pred": 0.0871
  },
  {
    "image_path": "./images/generated/image3.jpg",
    "pred": 0.9234
  }
]
```

### Output Fields

| Field        | Description                                                                |
| ------------ | -------------------------------------------------------------------------- |
| `image_path` | Path to the processed image                                                |
| `pred`       | Model sigmoid output representing estimated probability of AIGC generation |

The `pred` value ranges from **0 to 1**.

Higher values indicate stronger model confidence that the image is AI-generated.

For example:

```text
pred = 0.92
```

indicates a strong model prediction toward AIGC generation, while:

```text
pred = 0.08
```

indicates a strong prediction toward a non-generated/real image.

---

# 🏗️ Model Architecture

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
                                  │ Importance-Weighted Fusion │
                                  │            Head             │
                                  └──────────────┬──────────────┘
                                                 │
                                            Sigmoid
                                                 │
                                                 ▼
                                        AIGC Confidence
```

## 1. DINOv2 Branch

A pretrained **DINOv2 ViT-S/14** backbone provides high-level visual representations.

The backbone is frozen during training, allowing the detector to build a lightweight classification system on top of the pretrained representation.

## 2. NPR Residual Branch

The residual branch extracts image artifacts associated with reconstruction, interpolation, and resampling.

This branch is intended to capture lower-level signals that may complement the semantic representation produced by DINOv2.

## 3. Texture Branch

The texture branch extracts handcrafted statistical descriptors:

* **Local Binary Patterns (LBP)**
* **Gray-Level Co-occurrence Matrix (GLCM)**

These features provide additional information about local texture and spatial relationships between pixel intensities.

## 4. Importance-Weighted Fusion

The outputs of the three branches are projected into a common representation and combined through an importance-weighted fusion head.

The final representation is passed through a sigmoid function to produce the AIGC confidence score.

---

# 🧪 Training & Evaluation

The complete experimental pipeline is provided in:

```text
KatsuDemo.ipynb
```

The notebook contains:

* Dataset preparation
* Image preprocessing
* Feature extraction
* Model construction
* Training
* Evaluation
* Robustness analysis
* Error analysis
* Interactive demonstration
* Model checkpoint generation
* Command-line inference preparation

For normal inference and evaluation of new images, use:

```bash
python predict.py
```

rather than executing the entire notebook.

---

# 🛡️ Redistribution Robustness

A central design goal of Katsu is maintaining useful detection performance when images undergo transformations commonly introduced during online redistribution.

The approach therefore does not depend on:

* Embedded metadata
* C2PA credentials
* Original filenames
* Platform-specific provenance
* Invisible watermarks

Instead, the detector operates directly on the available image content.

The robustness experiments and analysis are included in `KatsuDemo.ipynb`.

---

# ⚙️ System Requirements

### Recommended

* **Python:** 3.10+
* **RAM:** 16 GB+
* **GPU:** NVIDIA CUDA-compatible GPU recommended
* **Storage:** Sufficient space for PyTorch, DINOv2, and dependencies
* **Internet:** Required for the first DINOv2 download unless the backbone is already cached

CPU inference may be possible but is expected to be substantially slower than GPU inference.

---

# 🧰 Development

Create a virtual environment:

```bash
python -m venv katsu-env
```

Activate it on macOS/Linux:

```bash
source katsu-env/bin/activate
```

On Windows:

```bash
katsu-env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📓 Notebook

`KatsuDemo.ipynb` serves as the complete research and experimentation record for the project.

It contains the development pipeline used to:

1. Prepare the dataset
2. Extract model features
3. Train the hybrid detector
4. Evaluate performance
5. Investigate robustness
6. Perform error analysis
7. Save the trained checkpoint
8. Demonstrate inference

The standalone `predict.py` script provides the reproducible inference interface for external evaluation.

---

# 📌 Reproducible Inference

A fresh user can reproduce inference using:

```bash
git clone https://github.com/YOUR-USERNAME/katsu-hybrid-aigc-detector.git

cd katsu-hybrid-aigc-detector

pip install -r requirements.txt

python predict.py ./test_images \
    --checkpoint ./hybrid_checkpoint.pt \
    --output ./predictions.json
```

The resulting `predictions.json` contains:

```json
[
  {
    "image_path": "test_images/example.jpg",
    "pred": 0.XXXX
  }
]
```

---

# 🙏 Acknowledgements

Katsu builds upon open-source research and software, including:

* **DINOv2** for pretrained visual representations
* **PyTorch** for model development and inference
* **Torchvision** for computer-vision utilities
* **scikit-image** for image and texture feature extraction

The project also uses the datasets and resources documented in `KatsuDemo.ipynb`.

---

# 🏆 Competition

Katsu was developed for:

**TikTok TechJam 2026**

The project focuses on robust, content-based detection of AI-generated images under realistic redistribution conditions.

---

## 📄 License

See the repository for licensing information.

---

## 👥 Project

**Katsu-Hybrid-AIGC-Detector**

A lightweight hybrid approach to AI-generated image detection, combining pretrained semantic representations with complementary image-artifact and texture signals.

**Built for robust AIGC detection in the real world.**



