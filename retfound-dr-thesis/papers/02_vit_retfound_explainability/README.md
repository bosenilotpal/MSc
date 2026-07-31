# Paper 2 — Lesion-Grounded ViT-RETFound Explainability

## Topic

IEEE-journal study of whether RETFound explanations align with IDRiD lesion
annotations and remain faithful to model predictions.

## Contents

- `planning/Plan_Explainability_ViT_RETFound.md` — journal research plan
- `notebooks/RETFound_IDRiD_Explainability_Journal_Kaggle.ipynb` — experiment
- `scripts/build_explainability_notebook.py` — reproducible notebook generator

## Kaggle inputs

1. **IDRiD (full segmentation):** [`aaryapatel98/indian-diabetic-retinopathy-image-dataset`](https://www.kaggle.com/datasets/aaryapatel98/indian-diabetic-retinopathy-image-dataset)  
   Contains `A. Segmentation` (MA/HE/EX/SE/OD `.tif` masks) and `B. Disease Grading` (official 413+103 CSVs).  
   Do **not** use `mohamedabdalkader/...-idrid` — that mirror has grades/captions only, no lesion masks.
2. RETFound CFP weights (`RETFound_cfp_weights.pth`)
3. Saved `M0_best.pt` and `M2_best.pt` (plus histories / `splits.json` if available)

The notebook exports paper-ready figures, CSV statistics, JSON summaries, and
a ZIP archive under `/kaggle/working/`.
