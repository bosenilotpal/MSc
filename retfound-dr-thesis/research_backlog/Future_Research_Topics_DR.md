# Future Research Topics — Diabetic Retinopathy (DR)

Ideas for follow-up papers building on the current work
(RETFound + LoRA + multi-scale fusion + ordinal/referable losses, with APTOS training and Messidor-2 external validation).

## Direct extensions of the current work
1. **Multi-dataset domain generalization** — train on APTOS, test on Messidor-2 / DDR / IDRID / EyePACS; study domain adaptation and test-time adaptation.
2. **Uncertainty-aware DR grading** — calibrated confidence, selective referral ("refer when uncertain").
3. **Fairness & subgroup analysis** — performance by camera, ethnicity proxies, image quality, left/right eye.
4. **Few-shot / label-efficient RETFound** — 1% / 5% / 10% labels with LoRA vs full fine-tune.
5. **Loss-scheduling for ordinal + referable objectives** — fix A3's QWK drop with adaptive weighting.
6. **Explainability for ViT-RETFound** — attention maps / Grad-CAM / concept-based explanations vs lesion annotations.

## Clinically high-value topics
7. **Referable-DR screening systems** — sensitivity/specificity at fixed operating points, cost–benefit of false negatives.
8. **Progression prediction** — predict worsening grade over time from longitudinal fundus (if data available).
9. **DR + DME joint modeling** — multi-task severity + macular edema.
10. **Image quality assessment + reject option** — don't grade ungradable images; compare vs forced prediction.

## Method / ML novelty
11. **Ordinal transformers for DR** — CORAL / CORN / ordinal embedding heads on RETFound.
12. **Multi-modal fundus + OCT** — fuse RETFound CFP with OCT features (if data accessible).
13. **Self-supervised continued pretraining** on target cameras before DR fine-tuning.
14. **Knowledge distillation** — RETFound teacher → lightweight student for edge / clinic devices.
15. **Active learning for DR annotation** — reduce labeling cost while preserving QWK.

## Most publishable next papers (practical ranking)
| Priority | Topic | Why it fits |
|----------|-------|-------------|
| 1 | Domain generalization + Messidor/DDR | External validation setup already exists |
| 2 | Uncertainty / selective referral | Strong clinical story, builds on referable AUROC |
| 3 | Few-shot LoRA vs full FT | Clean ablation paper; cheap to run |
| 4 | Fixing multi-objective loss trade-off (A2 vs A3) | Directly motivated by current ablation finding |
| 5 | Lightweight distilled DR grader | Deployment angle; complementary to foundation models |
