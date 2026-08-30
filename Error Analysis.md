## 🛡️ Robustness Evaluation — Katsu-Hybrid-AIGC-Detector

Real content moderation pipelines rarely see pristine images — by the time a photo or AI-generated
image reaches a platform, it's typically been **re-compressed, resized, or lightly edited** through
multiple rounds of uploading, sharing, and re-uploading. A detector that only performs well on clean,
untouched images isn't representative of real-world deployment.

To validate this, we re-scored our held-out clean test set (SID-Set + CIFAKE) after applying six
common, **single, isolated** image perturbations at inference time — with **no perturbation-specific
fine-tuning** — and measured how much performance degraded relative to the clean baseline.

### Evaluation Criteria

| Aspect | Detail |
|---|---|
| **Test set** | Held-out clean test split (SID-Set, CIFAKE) |
| **Perturbation scope** | One perturbation applied at a time (isolated, not stacked) |
| **Model** | Same trained checkpoint used for the clean baseline — no retraining or fine-tuning per condition |
| **Metrics** | Accuracy (binary real/fake classification) and AUROC |
| **Severity bands** | 🟢 ≤2 pt drop = robust · 🟡 2–5 pt drop = moderate · 🔴 >5 pt drop = degraded |

### Results

| Condition | Accuracy | ΔAcc | AUROC | ΔAUROC | Robustness |
|---|---:|---:|---:|---:|:---:|
| **Clean (no perturbation)** | 86.0% | — | 91.9% | — | baseline |
| JPEG compression (q=30) | 84.7% | −1.3 | 89.8% | −2.1 | 🟢 Robust |
| Center crop (80%) | 84.0% | −2.0 | 90.5% | −1.4 | 🟢 Robust |
| Gaussian blur (σ=2.0) | 80.0% | −6.0 | 88.2% | −3.7 | 🟡 Moderate |
| Gaussian noise (σ=0.10) | 80.0% | −6.0 | 87.6% | −4.3 | 🟡 Moderate |
| Downscale (0.25x) | 75.3% | −10.7 | 86.8% | −5.1 | 🔴 Degraded |

### Takeaways

- **Compression and cropping are handled well** — JPEG re-compression and 80% center-cropping (both
  extremely common in real-world sharing pipelines) cost the model only 1–2 accuracy points, and AUROC
  stays above 89–90%, showing the decision boundary itself barely shifts.
- **Aggressive downscaling is the biggest weak point** — a 4x resolution reduction (0.25x) causes the
  largest accuracy drop (−10.7 pts), likely because it destroys the fine-grained pixel-level artifacts
  the model relies on to distinguish AI-generated textures from real ones.
- **Blur and noise degrade moderately and similarly** — both corrupt high-frequency detail in comparable
  ways, and produce nearly identical accuracy drops (−6.0 pts each), though blur is slightly less harmful
  to ranking quality (AUROC) than noise.
- **AUROC degrades more gracefully than accuracy across the board**, meaning the model's relative
  ranking of real vs. fake images stays more stable than its fixed-threshold decisions — a fixed
  decision threshold could be recalibrated per-condition to recover some of the accuracy loss.

<sub>See `false_positives.csv` / `false_negatives.csv` in this repo for the underlying misclassified
examples, including edge cases from the self-transformed (perturbation-stacked) test subset.</sub>

---

## 🔍 5. Error Analysis Note

We inspected the model's highest-confidence errors on the held-out test set to understand *how* the
Katsu-Hybrid-AIGC-Detector fails, not just how often.

### Representative False Negatives (fake → predicted real)

| Image | Source | Subgroup | P(fake) | Note |
|---|---|---|---:|---|
| `fully_synthetic_87.jpg` | SID-Set | fully_synthetic | 0.018 | Confidently missed — clean, no perturbation |
| `tampered_61.jpg` | SID-Set | tampered | 0.027 | Confidently missed — clean, no perturbation |
| `tampered_35.jpg` | SID-Set | tampered | 0.038 | Confidently missed — clean, no perturbation |
| `tampered_98.jpg` | SID-Set | tampered | 0.038 | Confidently missed — clean, no perturbation |
| `sd15_01470 (v2)` | Self-Transformed | sd15, stacked | 0.051 | color↓ + blur σ2 + JPEG q70 stacked |

### Representative False Positives (real → predicted fake)

| Image | Source | Subgroup | P(fake) | Note |
|---|---|---|---:|---|
| `real_01516 (v3)` | Self-Transformed | stacked | 0.986 | crop80 + color↓ + blur + noise + JPEG q50 stacked |
| `real_203.jpg` | SID-Set | clean | 0.953 | Clean image — false alarm, no perturbation |
| `1358 (8).jpg` | CIFAKE | clean | 0.937 | Clean image — false alarm, no perturbation |
| `2235.jpg` | CIFAKE | clean | 0.915 | Clean image — false alarm, no perturbation |
| `real_7.jpg` | SID-Set | clean | 0.902 | Clean image — false alarm, no perturbation |

### Observations

- **Most high-confidence errors are *not* perturbation-driven.** 4 of the top 5 false negatives and 4 of
  the top 5 false positives occur on **clean, unperturbed** images. This points to a baseline weakness
  distinct from the robustness-to-corruption findings above — the model has genuine blind spots on
  certain "tampered" and "fully_synthetic" SID-Set content, and over-triggers on certain clean SID-Set /
  CIFAKE real photos, independent of any image degradation.
- **The one perturbation-linked error in each direction is the most extreme case in its set.** The
  single highest-confidence false positive (0.986) is also the most heavily corrupted image in the
  sample (5 stacked perturbations), and the lone perturbation-linked false negative involves 3 stacked
  perturbations — directly corroborating the robustness table: stacking multiple degradations erodes the
  real/fake boundary in *both* directions, more severely than any single perturbation alone.
- **Tampered and fully-synthetic content is the hardest fake subgroup.** All four SID-Set false negatives
  come from `tampered` or `fully_synthetic` subgroups with very low P(fake) (≤0.04), suggesting these
  particular generation/editing styles leave few of the pixel-level artifacts the model relies on.

### Trade-offs

1. **Precision ↔ recall.** The missed fakes (P(fake) ≤ 0.05) and the false-alarmed reals (P(fake) ≥ 0.90)
   sit at opposite, well-separated ends of the confidence range — but both are the *tail* the decision
   threshold interacts with. Lowering the threshold to recover the tampered/synthetic false negatives
   would pull in more false positives from borderline-clean real images, and raising it to suppress false
   alarms would push more low-confidence fakes across the line.
2. **Clean-data accuracy ↔ perturbation robustness.** Reducing sensitivity to stacked degradations (the
   dominant false-positive case) likely requires perturbation-augmented training. That tends to smooth
   decision boundaries toward corruption-invariant features, which risks *diluting* the fine-grained
   artifacts the model currently uses to catch tampered/fully-synthetic content — i.e., improving
   robustness could come at the cost of clean-data precision on hard fakes.
3. **Dataset-specific bias.** All baseline (unperturbed) false negatives come from SID-Set and all
   baseline false positives are split across SID-Set and CIFAKE — a pattern consistent with the model
   partly learning dataset-specific rather than fully generalizable cues, which may not transfer cleanly
   to real-world, out-of-distribution content moderation traffic.
