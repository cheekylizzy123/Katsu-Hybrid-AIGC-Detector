# Katsu-Hybrid-AIGC-Detector
Katsu is a lightweight, self-supervised-backbone hybrid model for detecting AI-generated images, built specifically to stay accurate after the image has been through real-world redistribution (recompression, resizing, cropping, noise). It combines a **frozen DINOv2-S/14 backbone** with two complementary, lightweight artifact-detection branches: an **NPR (upsampling-residual) branch** and a **texture-statistics (LBP/GLCM) branch**, fused through an **importance-weighted gating head**. Only **33,842 trainable parameters** sit on top of the frozen backbone. This model is built for Tik Tok's annual hackathon **TechJam 2026**.
<img width="1024" height="1059" alt="Katsu_Summary_Visual" src="https://github.com/user-attachments/assets/6d2cc0fd-172f-41be-a4e6-af7d3777ba73" />


## How This Solution Addresses the Problem Statement

Platforms like TikTok already auto-label AI-generated content via C2PA Content Credentials and invisible watermarks, but that provenance metadata is stripped the moment content is screenshotted, re-encoded, or reposted off-platform. Once that happens, there's no metadata left to check.

Katsu is a **content-based fallback layer** for exactly that scenario: it doesn't rely on any embedded signal, only the pixels themselves, and it's trained specifically to keep working after the kinds of transformations that strip metadata in the first place. That's the "redistribution robustness" this problem statement is testing — not just detecting AI-generated images in their original, clean form.


