# LoRA-Adapted RETFound with Multi-Scale Fusion and Ordinal Losses for Five-Class Diabetic Retinopathy Grading and External Validation

**Nilotpal Bose**  
Department of Computer Science and Engineering  
Email: *[update-email]*  

---

## Abstract

Adapting retinal foundation models for ordered DR severity remains challenging when flat cross-entropy ignores class imbalance and clinical ordinal structure. We compare **RETFound Baseline** (full fine-tuning with weighted cross-entropy) and **Enhanced RETFound** (LoRA adapters, multi-scale token fusion, and focal/ordinal/referable losses) for five-class diabetic retinopathy grading. On APTOS 2019 (test N = 550), Enhanced RETFound slightly improved quadratic weighted kappa (QWK 0.893 vs 0.888) and referable AUROC (0.987 vs 0.981), while the baseline retained higher accuracy (0.831 vs 0.811). On external Messidor-2 without fine-tuning, Enhanced RETFound improved QWK (0.505 vs 0.488) and referable AUROC (0.778 vs 0.725). Ablations show A2 (LoRA + multi-scale + focal) achieves the best APTOS QWK (0.895), whereas ordinal/referable auxiliaries mainly boost screening AUROC. Overall, Enhanced RETFound is comparable in-domain and stronger under external shift for ordinal and screening-oriented metrics.

## 1. Introduction

Diabetic retinopathy is a microvascular complication of diabetes and a major cause of preventable vision loss worldwide. Population screening relies on grading colour fundus photographs according to clinical severity scales such as the International Clinical Diabetic Retinopathy (ICDR) system, which orders disease from absent (grade 0) through mild, moderate, and severe non-proliferative disease to proliferative DR (grade 4). Because grading is labour-intensive and subject to inter-observer variability, deep learning has been widely explored for automated DR detection and staging.

Early systems were typically trained from scratch or from ImageNet-initialised convolutional networks on large labelled fundus corpora. More recently, foundation models pretrained with self-supervised learning on massive unlabelled medical image collections have emerged as a stronger starting point for downstream clinical tasks. **RETFound** is a retinal foundation model based on a Vision Transformer pretrained with masked autoencoding on approximately 1.6 million fundus and OCT images, and has demonstrated strong transfer to ocular and systemic prediction tasks with limited labels.

Despite this progress, adapting RETFound to DR severity grading still raises open questions. Standard full fine-tuning with flat cross-entropy treats grades as unordered classes and may under-emphasise rare mild and severe cases, as well as ordinal structure. Parameter-efficient fine-tuning methods such as LoRA, multi-scale feature aggregation, and imbalance-/ordinal-aware losses (e.g., focal loss and CORAL-style ordinal regression) offer a more clinically aligned adaptation recipe. In addition, strong in-domain scores on a single public dataset (e.g., APTOS 2019) do not guarantee robustness under camera and population shift, motivating external evaluation on Messidor-2.

**Objective.** This study compares RETFound Baseline against an Enhanced RETFound adaptation for five-class DR grading, with ablations and Messidor-2 external testing, and reports both accuracy-based and ordinal/screening metrics (notably QWK).

## 2. Materials and Methods

### 2.1 Dataset Description

**APTOS 2019 Blindness Detection** provides labelled macular-centred fundus photographs collected in clinical settings in India. Each image is annotated with an ICDR-aligned severity grade from 0 to 4. The dataset is publicly known for class imbalance (no-DR and moderate grades dominate; mild and severe grades are scarce), which makes macro-averaged and ordinal metrics especially informative. We used a stratified 70%/15%/15% train/validation/test split fixed in `splits.json` (test N = 550). Images were resized to 224x224, normalised with ImageNet statistics, and lightly augmented during training (flips, small rotations, mild colour jitter).

**Messidor-2** is used strictly as an **external test set** (no fine-tuning). It contains fundus examinations acquired under different devices and protocols from APTOS, and therefore probes domain generalisation. We evaluate five-class grading and a clinically actionable binary endpoint: **referable DR**, defined as grade >= 2 (moderate or worse), consistent with common screening practice.

### 2.2 Frameworks for Comparison

Two main frameworks share the same RETFound ViT-Large/16 colour-fundus pretrained backbone and the same APTOS split/preprocessing.

**RETFound Baseline.** The entire encoder is fully fine-tuned with a linear classification head and weighted cross-entropy. This matches the conventional transfer-learning recipe for foundation-model adaptation and serves as a strong reference.

**Enhanced RETFound.** To reduce catastrophic forgetting and focus learning on DR-relevant adaptation, the backbone is largely frozen and updated through **LoRA** adapters on attention `qkv` projections (~1.3% trainable parameters). Features are taken from multiple transformer depths (blocks 7, 15, and 23) and fused before classification, capturing both local lesion cues and global severity context. Training optimises a combination of **focal loss** for hard/rare classes, **CORAL ordinal loss** for ordered severity, and a binary **referable-DR** auxiliary head. Checkpoints for both frameworks were selected by best validation QWK.

**Ablations (A1-A3).** To isolate contributions, we trained three Enhanced variants for 15 epochs on the same split: **A1** LoRA + late features + focal; **A2** A1 + multi-scale fusion; **A3** A2 + ordinal + referable auxiliaries.

**Evaluation.** We report accuracy, macro-F1, QWK, referable accuracy/AUROC, confusion matrices, and ROC curves. Primary interpretation emphasises QWK for grading agreement and referable AUROC for screening utility.

## 3. Results

### 3.1 Main Comparison on APTOS and Messidor-2

| Model | Set | Acc | Macro-F1 | QWK | Ref Acc | Ref AUROC |
|-------|-----|-----|----------|-----|---------|-----------|
| RETFound Baseline | APTOS | 0.831 | 0.671 | 0.888 | 0.929 | 0.981 |
| Enhanced RETFound | APTOS | 0.811 | 0.683 | 0.893 | 0.935 | 0.987 |
| RETFound Baseline | Messidor-2 | 0.612 | 0.330 | 0.488 | 0.807 | 0.725 |
| Enhanced RETFound | Messidor-2 | 0.593 | 0.349 | 0.505 | 0.802 | 0.778 |

On the APTOS held-out test set, Enhanced RETFound yields a small gain in QWK (+0.005) and referable AUROC (+0.006), while Baseline remains higher in overall accuracy (+0.020). This indicates a trade-off: the enhancement shifts performance toward ordinal agreement and screening-oriented discrimination instead of maximizing top-1 accuracy.

On external Messidor-2, Enhanced RETFound improves QWK (+0.017) and referable AUROC (+0.053), suggesting better robustness under domain shift for clinically relevant endpoints.

### 3.2 Validation QWK Curves

Validation QWK trends show rapid early gains followed by stabilization for both main models. Checkpoints were selected by maximum validation QWK to reduce the chance of overfit late-epoch selection.

For ablations, QWK trajectories support that A2's best test QWK is associated with a stable validation trend rather than isolated noisy peaks.

### 3.3 Ablation Study

All ablation variants were trained on the same APTOS split for 15 epochs and evaluated on the same test set.

| Variant | What is enabled | Acc | QWK | Ref AUROC |
|---------|-----------------|-----|-----|-----------|
| RETFound Baseline | Full fine-tuning + CE | 0.831 | 0.888 | 0.981 |
| Enhanced RETFound | Full Enhanced (loaded primary checkpoint) | 0.811 | 0.893 | 0.987 |
| A1 | LoRA + late features + focal | 0.738 | 0.892 | 0.985 |
| A2 | A1 + multi-scale fusion | 0.765 | 0.895 | 0.987 |
| A3 | A2 + ordinal + referable losses | 0.764 | 0.881 | 0.988 |

Key observations:

1. LoRA + focal (A1) is nearly sufficient for Enhanced-level QWK.
2. Multi-scale fusion (A2) provides the strongest grading gain (highest QWK).
3. Ordinal/referable auxiliaries (A3) increase referable AUROC but reduce QWK under the short 15-epoch schedule.

## 4. Discussion

The results show that model quality depends on the target clinical objective. If the objective is strict five-class agreement, A2 offers the strongest performance. If the objective prioritizes referable screening discrimination, Enhanced configurations with auxiliary objectives provide benefits, especially under external shift.

The in-domain improvements are modest, but external gains are more meaningful, reinforcing the importance of out-of-distribution validation before deployment claims.

## 5. Limitations

- Single primary training dataset (APTOS) limits diversity of training acquisition conditions.
- External evaluation is restricted to Messidor-2.
- Ablations use a shorter schedule (15 epochs), which may under-represent fully converged multi-objective behavior.
- Statistical significance tests are not fully expanded in this manuscript version.

## 6. Conclusion

This work compared standard RETFound fine-tuning with a clinically motivated Enhanced RETFound adaptation for DR severity grading. On APTOS, Enhanced RETFound is comparable to the baseline with small gains in QWK and referable AUROC but lower accuracy. External testing on Messidor-2 provides stronger support: Enhanced RETFound improves both QWK and referable AUROC under domain shift. These findings suggest that foundation-model adaptation for DR should be evaluated with ordinal and clinically meaningful endpoints, not accuracy alone, and always with external validation.

## References

[1] Zhou, Y., Chia, M.A., Wagner, S.K., et al. (2023). A foundation model for generalizable disease detection from retinal images. *Nature*, 622, 156-163.  
[2] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR*.  
[3] He, K., Chen, X., Xie, S., et al. (2022). Masked autoencoders are scalable vision learners. *CVPR*, 16000-16009.  
[4] Hu, E.J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR*.  
[5] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal loss for dense object detection. *ICCV*, 2980-2988.  
[6] Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank consistent ordinal regression for neural networks with application to age estimation. *Pattern Recognition Letters*, 140, 325-331.  
[7] Wilkinson, C.P., Ferris, F.L., Klein, R.E., et al. (2003). Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales. *Ophthalmology*, 110(9), 1677-1682.  
[8] Gulshan, V., Peng, L., Coram, M., et al. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. *JAMA*, 316(22), 2402-2410.  
[9] Decenciere, E., Zhang, X., Cazuguel, G., et al. (2014). Feedback on a publicly distributed image database: the Messidor database. *Image Analysis & Stereology*, 33(3), 231-234.  
[10] Karthik, Maggie, & Dane, S. (2019). APTOS 2019 Blindness Detection. Kaggle. https://www.kaggle.com/competitions/aptos2019-blindness-detection  
[11] Krause, J., Gulshan, V., Rahimy, E., et al. (2018). Grader variability and the importance of reference standards for evaluating machine learning models for diabetic retinopathy. *Ophthalmology*, 125(8), 1264-1272.  
[12] Ting, D.S.W., Pasquale, L.R., Peng, L., et al. (2019). Artificial intelligence and deep learning in ophthalmology. *British Journal of Ophthalmology*, 103(2), 167-175.  
[13] Li, T., Gao, Y., Wang, K., Guo, S., Liu, H., & Kang, H. (2019). Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening. *Information Sciences*, 501, 511-522.  
[14] Cohen, J. (1968). Weighted kappa: Nominal scale agreement with provision for scaled disagreement or partial credit. *Psychological Bulletin*, 70(4), 213-220.  
[15] Bommasani, R., Hudson, D.A., Adeli, E., et al. (2021). On the opportunities and risks of foundation models. *arXiv:2108.07258*.  
