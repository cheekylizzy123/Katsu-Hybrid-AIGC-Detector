# Katsu — Hybrid AIGC Detector (Streamlit)

**Try it live → https://katsu-demogit-ix9aua9mfr9fv4akep7dku.streamlit.app/**

## What the demo shows

- **Classification decision** — real vs. fake, with a confidence score
- **Branch weight contribution** — how much each of the three detection
  branches (DINOv2, NPR, texture) influenced the final decision
- **Spatial attribution maps** — heatmaps showing *where* in the image
  the DINOv2 and NPR branches found the strongest signal
- **Texture features** — the raw LBP/GLCM statistics the texture branch
  extracted from the image
- **Robustness testing** — optional toggles (blur, noise, JPEG
  compression, brightness) to see how predictions hold up under common
  image transformations

## Running it locally instead

If you'd rather run the demo yourself rather than use the live link:

```bash
cd katsu-streamlit-demo
pip install -r requirements.txt
streamlit run app.py
```

This loads `hybrid_checkpoint.pt` (included in this folder) and starts
a local server — Streamlit will print a `localhost` URL to open in your
browser. The DINOv2 backbone downloads automatically on first run.

