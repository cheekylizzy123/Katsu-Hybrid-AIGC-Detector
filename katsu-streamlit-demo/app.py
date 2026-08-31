# ==========================================
# Katsu — AIGC Detection (Streamlit Community Cloud entry point)
# Loads the trained model from a checkpoint file at startup.
# No dataset downloads, no training — inference only.
# ==========================================
import os, time, io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter

from predict import (
    DEVICE, dino_preprocess, images_to_raw_batch, compute_npr_residual_batch,
    extract_texture_features, load_models,
)

BRANCH_ORDER = ["dino", "npr", "texture"]  # matches training order exactly

# ---------- Locate the checkpoint ----------
# checkpoint file committed directly to the repo (via git-lfs).
CHECKPOINT_PATH = st.secrets.get("CHECKPOINT_PATH", "hybrid_checkpoint.pt")
CHECKPOINT_REPO_ID = st.secrets.get("CHECKPOINT_REPO_ID", None)
CHECKPOINT_FILENAME = st.secrets.get("CHECKPOINT_FILENAME", "hybrid_checkpoint.pt")


@st.cache_resource(show_spinner="Loading Katsu model (first load only)...")
def get_models():
    checkpoint_path = CHECKPOINT_PATH
    if CHECKPOINT_REPO_ID:
        from huggingface_hub import hf_hub_download
        checkpoint_path = hf_hub_download(repo_id=CHECKPOINT_REPO_ID, filename=CHECKPOINT_FILENAME)
    return load_models(checkpoint_path)


_models = get_models()
dinov2 = _models["dinov2"]
residual_encoder = _models["residual_encoder"]
texture_proj = _models["texture_proj"]
fusion_head = _models["fusion_head"]


def hybrid_forward(pil_imgs):
    dino_batch = torch.stack([dino_preprocess(img) for img in pil_imgs]).to(DEVICE)
    with torch.no_grad():
        dino_feats = dinov2(dino_batch)

        raw_batch = images_to_raw_batch(pil_imgs)
        residuals = compute_npr_residual_batch(raw_batch)
        npr_feats = residual_encoder(residuals)

        tex_vecs = torch.tensor(np.stack([extract_texture_features(img) for img in pil_imgs])).to(DEVICE)
        tex_feats = texture_proj(tex_vecs)

        logits, weights = fusion_head([dino_feats, npr_feats, tex_feats])
    return logits, weights


# ---------- Real spatial signal, self-contained ----------

def get_dino_spatial_heatmap(pil_img, target_size):
    x = dino_preprocess(pil_img).unsqueeze(0).to(DEVICE)
    try:
        with torch.no_grad():
            feats = dinov2.get_intermediate_layers(x, n=1, reshape=True)[0]
        act = feats.squeeze(0).mean(dim=0).cpu().numpy()
    except (AttributeError, TypeError) as e:
        st.write(f"DINOv2 spatial extraction unavailable ({e}); using uniform fallback map.")
        act = np.ones((16, 16), dtype=np.float32) * 0.5
    act = np.maximum(act, 0)
    if act.max() > 0:
        act = act / act.max()
    return cv2.resize(act, target_size)


def get_npr_spatial_heatmap(pil_img, target_size, scale=0.5):
    arr = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
    t = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    down = F.interpolate(t, scale_factor=scale, mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=(224, 224), mode="bilinear", align_corners=False)
    residual = (t - up).squeeze(0)
    act = residual.abs().mean(dim=0).cpu().numpy()
    if act.max() > 0:
        act = act / act.max()
    return cv2.resize(act, target_size)


def overlay_heatmap(image_np, heatmap_2d, alpha=0.45):
    h, w, _ = image_np.shape
    if heatmap_2d.shape != (h, w):
        heatmap_2d = cv2.resize(heatmap_2d, (w, h))
    saliency_uint8 = np.uint8(255 * heatmap_2d)
    colored = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_np, 1 - alpha, colored_rgb, alpha, 0)

    active_mask = saliency_uint8 > 120
    active_ratio = (np.sum(active_mask) / (h * w)) * 100
    if np.any(active_mask):
        peak_y, peak_x = np.unravel_index(np.argmax(saliency_uint8), (h, w))
        v = "top" if peak_y < h / 3 else ("bottom" if peak_y > 2 * h / 3 else "center")
        hpos = "left" if peak_x < w / 3 else ("right" if peak_x > 2 * w / 3 else "center")
        loc_str = f"{v}-{hpos}" if v != hpos else v
    else:
        loc_str = "diffuse/broad region"
    return overlay, active_ratio, loc_str


def texture_feature_bar(pil_img):
    feats = extract_texture_features(pil_img)
    labels = [f"LBP_{i}" for i in range(10)] + [f"GLCM_{i}" for i in range(16)]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(len(feats)), feats, color="teal")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_title("Texture branch — raw LBP/GLCM feature values")
    plt.tight_layout()
    return fig


# ---------- Transform application ----------

def apply_selected_transforms(pil_img, selected_transforms):
    if not selected_transforms or pil_img is None:
        return pil_img
    transformed = pil_img.copy()
    if "Gaussian Blur" in selected_transforms:
        transformed = transformed.filter(ImageFilter.GaussianBlur(radius=2))
    if "Add Gaussian Noise" in selected_transforms:
        np_img = np.array(transformed).astype(np.float32)
        noise = np.random.normal(0, 12, np_img.shape)
        transformed = Image.fromarray(np.clip(np_img + noise, 0, 255).astype(np.uint8))
    if "JPEG Compression" in selected_transforms:
        buf = io.BytesIO()
        transformed.save(buf, format="JPEG", quality=35)
        buf.seek(0)
        transformed = Image.open(buf).convert("RGB")
    if "Brightness Adjustment" in selected_transforms:
        transformed = ImageEnhance.Brightness(transformed).enhance(1.25)
    return transformed


# ---------- Inference ----------

def run_hybrid_inference(pil_img, selected_transforms):
    start_time = time.time()
    processed_img = apply_selected_transforms(pil_img, selected_transforms).convert("RGB")
    img_np = np.array(processed_img)
    target_size = (processed_img.width, processed_img.height)

    logits, weights = hybrid_forward([processed_img])
    prob = torch.sigmoid(logits).item()
    b_weights = weights.detach().cpu().numpy().squeeze(0)

    latency_ms = f"{int((time.time() - start_time) * 1000)} ms"
    weight_dict = {name: float(w) for name, w in zip(BRANCH_ORDER, b_weights)}
    verdict_label = f"Classified as {'Fake' if prob > 0.5 else 'Real'} — {prob*100:.1f}% confidence AI-generated"

    dino_map = get_dino_spatial_heatmap(processed_img, target_size)
    dino_overlay, dino_ratio, dino_loc = overlay_heatmap(img_np, dino_map)

    npr_map = get_npr_spatial_heatmap(processed_img, target_size)
    npr_overlay, npr_ratio, npr_loc = overlay_heatmap(img_np, npr_map)

    texture_fig = texture_feature_bar(processed_img)

    dino_exp = f"DINOv2 branch: peak activation in {dino_loc} region, covering {dino_ratio:.1f}% of the frame."
    npr_exp = f"NPR branch: highest reconstruction residual in {npr_loc} region, covering {npr_ratio:.1f}% of the frame."

    return {
        "processed_img": processed_img, "verdict": verdict_label, "prob": float(prob),
        "latency": latency_ms, "weights": weight_dict,
        "dino_overlay": dino_overlay, "dino_exp": dino_exp,
        "npr_overlay": npr_overlay, "npr_exp": npr_exp,
        "texture_fig": texture_fig,
    }


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Katsu — AIGC Detection", page_icon="🔎", layout="wide")

st.markdown(
    """
    <div style="text-align:center;">
    <h1 style="color:#7c3aed; margin-bottom:0;">🔎 Katsu</h1>
    <p style="color:#6b7280; margin-top:4px;">Hybrid AIGC Detector — Forensic Audit Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1, 2])

with left:
    uploaded_file = st.file_uploader("Upload Target Image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    selected_transforms = st.multiselect(
        "Apply Transformations",
        ["Gaussian Blur", "Add Gaussian Noise", "JPEG Compression", "Brightness Adjustment"],
    )
    analyze_clicked = st.button("Analyze Image", type="primary", use_container_width=True)
    latency_placeholder = st.empty()
    latency_placeholder.text_input("Inference Latency", value="0 ms", disabled=True)

with right:
    preview_placeholder = st.empty()
    verdict_placeholder = st.empty()
    prob_placeholder = st.empty()
    weights_placeholder = st.empty()

st.markdown("---")

col_dino, col_npr, col_tex = st.columns(3)

if analyze_clicked:
    if uploaded_file is None:
        with right:
            verdict_placeholder.warning("No image uploaded")
    else:
        pil_img = Image.open(uploaded_file).convert("RGB")
        with st.spinner("Running inference..."):
            result = run_hybrid_inference(pil_img, selected_transforms)

        latency_placeholder.text_input("Inference Latency", value=result["latency"], disabled=True)
        with right:
            preview_placeholder.image(result["processed_img"], caption="Model Input Preview", use_container_width=True)
            verdict_placeholder.text_input("Classification Decision", value=result["verdict"], disabled=True)
            prob_placeholder.slider("Fake Probability", 0.0, 1.0, value=result["prob"], disabled=True)
            weights_placeholder.bar_chart(result["weights"])

        with col_dino:
            st.image(result["dino_overlay"], caption="DINOv2 — spatial attention", use_container_width=True)
            st.text_area("DINOv2 analysis", value=result["dino_exp"], height=70, disabled=True)
        with col_npr:
            st.image(result["npr_overlay"], caption="NPR — residual magnitude", use_container_width=True)
            st.text_area("NPR analysis", value=result["npr_exp"], height=70, disabled=True)
        with col_tex:
            st.pyplot(result["texture_fig"])
