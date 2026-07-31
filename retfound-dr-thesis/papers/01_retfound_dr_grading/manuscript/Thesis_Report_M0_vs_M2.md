# LoRA-Adapted RETFound with Multi-Scale Fusion and Ordinal Losses for Five-Class Diabetic Retinopathy Grading and External Validation

## Abstract

Adapting retinal foundation models for ordered DR severity remains challenging when flat cross-entropy ignores class imbalance and clinical ordinal structure. We compare **RETFound Baseline** (full fine-tuning with weighted cross-entropy) and **Enhanced RETFound** (LoRA adapters, multi-scale token fusion, and focal/ordinal/referable losses) for five-class diabetic retinopathy grading. On APTOS 2019 (test N = 550), Enhanced RETFound slightly improved quadratic weighted kappa (QWK 0.893 vs 0.888) and referable AUROC (0.987 vs 0.981), while the baseline retained higher accuracy (0.831 vs 0.811). On external Messidor-2 without fine-tuning, Enhanced RETFound improved QWK (0.505 vs 0.488) and referable AUROC (0.778 vs 0.725). Ablations show A2 (LoRA + multi-scale + focal) achieves the best APTOS QWK (0.895), whereas ordinal/referable auxiliaries mainly boost screening AUROC. Overall, Enhanced RETFound is comparable in-domain and stronger under external shift for ordinal/screening metrics.

---

## 1. Introduction

Diabetic retinopathy is a microvascular complication of diabetes and a major cause of preventable vision loss worldwide. Population screening relies on grading colour fundus photographs according to clinical severity scales such as the International Clinical Diabetic Retinopathy (ICDR) system, which orders disease from absent (grade 0) through mild, moderate, and severe non-proliferative disease to proliferative DR (grade 4). Because grading is labour-intensive and subject to inter-observer variability, deep learning has been widely explored for automated DR detection and staging.

Early systems were typically trained from scratch or from ImageNet-initialised convolutional networks on large labelled fundus corpora. More recently, foundation models pretrained with self-supervised learning on massive unlabelled medical image collections have emerged as a stronger starting point for downstream clinical tasks. **RETFound** is a retinal foundation model based on a Vision Transformer pretrained with masked autoencoding on approximately 1.6 million fundus and OCT images, and has demonstrated strong transfer to ocular and systemic prediction tasks with limited labels.

Despite this progress, adapting RETFound to DR severity grading still raises open questions. Standard full fine-tuning with flat cross-entropy treats grades as unordered classes and may under-emphasise rare mild/severe cases and ordinal structure. Parameter-efficient fine-tuning methods such as LoRA, multi-scale feature aggregation, and imbalance-/ordinal-aware losses (e.g., focal loss and CORAL-style ordinal regression) offer a more clinically aligned adaptation recipe. In addition, strong in-domain scores on a single public dataset (e.g., APTOS 2019) do not guarantee robustness under camera and population shift, motivating external evaluation on Messidor-2.

**Objective.** This study compares RETFound Baseline against an Enhanced RETFound adaptation for five-class DR grading, with ablations and Messidor-2 external testing, and reports both accuracy-based and ordinal/screening metrics (notably QWK).

---

## 2. Materials and Method

### 2.1 Dataset Description

**APTOS 2019 Blindness Detection** provides labelled macular-centred fundus photographs collected in clinical settings in India. Each image is annotated with an ICDR-aligned severity grade from 0 to 4. The dataset is publicly known for class imbalance (no-DR and moderate grades dominate; mild and severe grades are scarce), which makes macro-averaged and ordinal metrics especially informative. We used a stratified 70%/15%/15% train/validation/test split fixed in `splits.json` (test N = 550). Images were resized to 224×224, normalised with ImageNet statistics, and lightly augmented during training (flips, small rotations, mild colour jitter).

**Messidor-2** is used strictly as an **external test set** (no fine-tuning). It contains fundus examinations acquired under different devices and protocols from APTOS, and therefore probes domain generalisation. We evaluate five-class grading and a clinically actionable binary endpoint: **referable DR**, defined as grade ≥ 2 (moderate or worse), consistent with common screening practice.

### 2.2 The Frameworks for Comparison

Two main frameworks share the same RETFound ViT-Large/16 colour-fundus pretrained backbone and the same APTOS split/preprocessing.

**RETFound Baseline.** The entire encoder is fully fine-tuned with a linear classification head and weighted cross-entropy. This matches the conventional transfer-learning recipe for foundation-model adaptation and serves as a strong reference.

**Enhanced RETFound.** To reduce catastrophic forgetting and focus learning on DR-relevant adaptation, the backbone is largely frozen and updated through **LoRA** adapters on attention `qkv` projections (~1.3% trainable parameters). Features are taken from multiple transformer depths (blocks 7, 15, and 23) and fused before classification, capturing both local lesion cues and global severity context. Training optimises a combination of **focal loss** for hard/rare classes, **CORAL ordinal loss** for ordered severity, and a binary **referable-DR** auxiliary head. Checkpoints for both frameworks were selected by best validation QWK.

**Ablations (A1–A3).** To isolate contributions, we trained three Enhanced variants for 15 epochs on the same split: **A1** LoRA + late features + focal; **A2** A1 + multi-scale fusion; **A3** A2 + ordinal + referable auxiliaries.

**Evaluation.** We report accuracy, macro-F1, QWK, referable accuracy/AUROC, confusion matrices, and ROC curves. Primary interpretation emphasises QWK for grading agreement and referable AUROC for screening utility.

---

## 3. Results

### 3.1 Main comparison on APTOS

| Model | Set | Acc | Macro-F1 | QWK | Ref Acc | Ref AUROC |
|-------|-----|-----|----------|-----|---------|-----------|
| RETFound Baseline | APTOS | 0.831 | 0.671 | 0.888 | 0.929 | 0.981 |
| Enhanced RETFound | APTOS | 0.811 | 0.683 | 0.893 | 0.935 | 0.987 |
| RETFound Baseline | Messidor-2 | 0.612 | 0.330 | 0.488 | 0.807 | 0.725 |
| Enhanced RETFound | Messidor-2 | 0.593 | 0.349 | 0.505 | 0.802 | 0.778 |

On the APTOS held-out test set, Enhanced RETFound yields a small gain in QWK (+0.005) and referable AUROC (+0.006), while Baseline remains higher in overall accuracy (+0.020). This pattern indicates that the enhancement does not uniformly dominate all metrics; instead, it shifts performance toward ordinal agreement and screening-oriented discrimination. Because APTOS gains are modest, external validation and ablations are essential to judge whether the Enhanced design is practically useful.

*(Figures 1–2: APTOS metric bars; internal vs external QWK/AUROC.)*

#### 3.1.1 Validation QWK curves

Figure 3 shows validation QWK versus training epoch for RETFound Baseline and Enhanced RETFound. Both models improve rapidly early, then plateau; checkpoints were selected by maximum validation QWK. Figure 4 shows the same trajectories for ablations A1–A3 (15 epochs), confirming that A2’s test QWK is consistent with a stable validation trend.

*(Figures 3–4: main and ablation validation QWK curves.)*

### 3.2 External validation on Messidor-2

External validation tests whether models trained on APTOS remain useful when imaging devices, populations, and acquisition protocols change. Messidor-2 was therefore evaluated **without any additional fine-tuning**, using the same preprocessing and the APTOS-selected checkpoints.

**Overall external behaviour.** Absolute performance falls for both models relative to APTOS (Baseline QWK 0.888 → 0.488; Enhanced QWK 0.893 → 0.505). Such degradation is expected under domain shift and should not be interpreted as training failure. The scientifically relevant question is the **relative** gap between frameworks on the external set.

**Where Enhanced RETFound helps externally.** Enhanced RETFound improves Messidor-2 QWK from 0.488 to 0.505 (+0.017) and referable AUROC from 0.725 to 0.778 (+0.053). The referable-AUROC gain is substantially larger than the corresponding APTOS gap and is the strongest evidence that clinically oriented adaptation improves cross-dataset screening behaviour. Macro-F1 also favours Enhanced (0.349 vs 0.330). Baseline retains a small edge in raw accuracy (0.612 vs 0.593) and referable accuracy (0.807 vs 0.802), again showing that accuracy alone would understate Enhanced’s external benefit.

**Why external metrics matter clinically.** In screening workflows, the priority is often to rank or detect eyes that need referral (moderate or worse DR), not merely to maximise exact five-class accuracy on the training distribution. The Messidor-2 referable ROC comparison supports Enhanced RETFound in this role: its ROC lies above Baseline across operating points, consistent with the AUROC increase.

**Error-structure observation.** External confusion matrices show broader grade confusion than on APTOS for both models, reflecting harder domain transfer. Nevertheless, Enhanced’s better QWK implies that remaining errors are, on average, more ordinal-consistent (closer grades) than Baseline’s, which is preferable when severity is an ordered clinical construct.

*(Figures 5–6: Messidor-2 confusion matrices and referable ROC.)*

### 3.3 Confusion matrices and ROC analysis (APTOS)

APTOS confusion matrices show both models are excellent on grade 0 (no DR). Enhanced RETFound improves recall for mild (grade 1) and severe (grade 3) disease relative to Baseline, at the cost of more confusion involving moderate disease (grade 2). This is consistent with adjacent-grade ordinal errors, which QWK penalises less harshly than flat accuracy. Referable ROC curves remain strong for both models on APTOS (AUROC > 0.98), so in-domain screening discrimination is already near ceiling; the more discriminative ROC comparison appears on Messidor-2 (Section 3.2).

### 3.4 Ablation study

Ablations answer *which design choices inside Enhanced RETFound actually matter*. All ablation variants were trained on the **same APTOS split** for 15 epochs, with checkpoints selected by validation QWK, and then evaluated on the APTOS test set.

| Variant | What is enabled | Acc | QWK | Ref AUROC |
|---------|-----------------|-----|-----|-----------|
| RETFound Baseline | Full fine-tuning + CE | 0.831 | 0.888 | 0.981 |
| Enhanced RETFound | Full Enhanced (loaded primary checkpoint) | 0.811 | 0.893 | 0.987 |
| **A1** | LoRA + late features + focal | 0.738 | 0.892 | 0.985 |
| **A2** | A1 + multi-scale fusion | 0.765 | **0.895** | 0.987 |
| **A3** | A2 + ordinal + referable losses | 0.764 | 0.881 | **0.988** |

**A1 — LoRA + focal only.**  
A1 already reaches QWK 0.892, nearly matching the full Enhanced checkpoint (0.893). This shows that **parameter-efficient adaptation of RETFound**, combined with class-imbalance-aware focal loss, accounts for most of the ordinal grading signal. Accuracy is lower than Baseline (0.738 vs 0.831), indicating A1 redistributes errors rather than simply fitting majority classes. In other words, LoRA is not a weak substitute here; it is the core mechanism of the Enhanced framework.

**A2 — adding multi-scale fusion.**  
Adding multi-scale token fusion improves QWK to **0.895** (best among all variants) and raises accuracy from 0.738 to 0.765 versus A1. Referable AUROC also matches the Enhanced model (0.987). This supports the hypothesis that DR grading benefits from combining mid-level local cues (microaneurysms/exudates) with deeper global context (overall severity). Among the Enhanced family, **A2 is the strongest pure grading recipe**.

**A3 — adding ordinal and referable auxiliaries.**  
A3 attains the highest referable AUROC (0.988) but **reduces QWK to 0.881**, below both A2 and Baseline. Two interpretations are compatible with the data: (i) the extra loss terms emphasise screening separation (referable vs non-referable) more than fine-grained ordinal agreement; (ii) under a short 15-epoch ablation schedule, the multi-objective loss may be under-optimised relative to A2. Practically, A3 is attractive if referable detection is the deployment goal, whereas A2 is preferable if five-class QWK is the primary endpoint. The primary Enhanced model, trained longer with the full objective, sits between these behaviours (QWK 0.893, AUROC 0.987).

**Summary of ablation implications.**  
1. LoRA + focal (A1) is necessary and nearly sufficient for Enhanced-level QWK.  
2. Multi-scale fusion (A2) provides the clearest grading gain.  
3. Ordinal/referable auxiliaries (A3) help screening AUROC but can hurt QWK if not carefully scheduled/weighted.  
Therefore, the Enhanced design is justified as a configurable clinical adaptation stack rather than a single monolithic architecture.

*(Figure 8: ablation QWK / referable AUROC summary.)*

---

## 4. Conclusion

This work compared standard RETFound fine-tuning with a clinically motivated Enhanced RETFound recipe for DR severity grading. On APTOS, the enhanced model is **comparable** to the baseline, with small gains in QWK and referable AUROC but lower accuracy. **External Messidor-2 testing provides stronger support**: Enhanced RETFound improves QWK and, especially, referable AUROC under domain shift, which matters for screening-style deployment. Ablations show that LoRA + focal adaptation already recovers most Enhanced-level QWK, **multi-scale fusion (A2)** gives the best grading agreement, and ordinal/referable auxiliaries (A3) mainly boost screening AUROC. Foundation-model adaptation for DR should therefore be judged with ordinal and clinical endpoints—not accuracy alone—and external validation is essential before claiming practical benefit.

---

## 5. References

[1] Zhou, Y., Chia, M.A., Wagner, S.K., et al. (2023). A foundation model for generalizable disease detection from retinal images. *Nature*, 622, 156–163.

[2] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An image is worth 16×16 words: Transformers for image recognition at scale. *ICLR*.

[3] He, K., Chen, X., Xie, S., et al. (2022). Masked autoencoders are scalable vision learners. *CVPR*, 16000–16009.

[4] Hu, E.J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR*.

[5] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *ICCV*, 2980–2988.

[6] Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank consistent ordinal regression for neural networks with application to age estimation. *Pattern Recognition Letters*, 140, 325–331.

[7] Wilkinson, C.P., Ferris, F.L., Klein, R.E., et al. (2003). Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales. *Ophthalmology*, 110(9), 1677–1682.

[8] Gulshan, V., Peng, L., Coram, M., et al. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. *JAMA*, 316(22), 2402–2410.

[9] Decencière, E., Zhang, X., Cazuguel, G., et al. (2014). Feedback on a publicly distributed image database: the Messidor database. *Image Analysis & Stereology*, 33(3), 231–234.

[10] Karthik, Maggie, & Dane, S. (2019). APTOS 2019 Blindness Detection. Kaggle. https://www.kaggle.com/competitions/aptos2019-blindness-detection

[11] Krause, J., Gulshan, V., Rahimy, E., et al. (2018). Grader variability and the importance of reference standards for evaluating machine learning models for diabetic retinopathy. *Ophthalmology*, 125(8), 1264–1272.

[12] Ting, D.S.W., Pasquale, L.R., Peng, L., et al. (2019). Artificial intelligence and deep learning in ophthalmology. *British Journal of Ophthalmology*, 103(2), 167–175.

[13] Li, T., Gao, Y., Wang, K., Guo, S., Liu, H., & Kang, H. (2019). Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening. *Information Sciences*, 501, 511–522.

[14] Cohen, J. (1968). Weighted kappa: Nominal scale agreement with provision for scaled disagreement or partial credit. *Psychological Bulletin*, 70(4), 213–220.

[15] Bommasani, R., Hudson, D.A., Adeli, E., et al. (2021). On the opportunities and risks of foundation models. *arXiv:2108.07258*.
