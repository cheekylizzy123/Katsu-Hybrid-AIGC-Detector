# Katsu — Hybrid AIGC Detector (Streamlit)

Upload an image and Katsu classifies it as real or AI-generated, using a
DINOv2 backbone plus two complementary artifact-detection branches (an NPR
residual signal and handcrafted LBP/GLCM texture statistics), fused with an
importance-weighted gate. Optional test-time transforms (blur, noise, JPEG
compression, brightness) let you check robustness.

Inference-only — loads a pretrained checkpoint (`hybrid_checkpoint.pt`) at
startup. No training, no dataset downloads.

## Deploying on Streamlit Community Cloud

1. Push this folder (including `hybrid_checkpoint.pt`, tracked via git-lfs)
   to a GitHub repo.
2. Go to share.streamlit.io → "New app" → point it at the repo, branch,
   and `app.py`.
3. If you'd rather not commit the checkpoint binary to git, skip it and
   instead set these in the app's Settings → Secrets:
   ```
   CHECKPOINT_REPO_ID = "your-username/katsu-weights"
   ```
   (a separate Hugging Face model repo hosting just the .pt file) —
   `app.py` will download it automatically at startup via `hf_hub_download`.
