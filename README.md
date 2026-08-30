# Katsu-Hybrid-AIGC-Detector
A lightweight hybrid AI-generated image detector built on a **frozen DINOv2-S/14 backbone** as the sole vision encoder, combined with two complementary artifact-detection branches: an **NPR (upsampling-residual) branch** and a **texture-statistics (LBP/GLCM) branch** fused through an **importance-weighted gating head**.
