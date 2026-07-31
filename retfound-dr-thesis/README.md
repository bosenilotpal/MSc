# RETFound Diabetic Retinopathy Research

The workspace is organized by paper and experiment topic.

## Structure

```text
papers/
├── 01_retfound_dr_grading/
│   ├── manuscript/   # Word, Markdown, and arXiv/LaTeX manuscripts
│   ├── notebooks/    # Baseline, Enhanced, ablation, and external-validation runs
│   ├── figures/      # Paper-ready plots
│   ├── scripts/      # Report and document builders
│   └── build/        # Generated LaTeX artifacts/PDF
└── 02_vit_retfound_explainability/
    ├── planning/     # IEEE journal research plan
    ├── notebooks/    # IDRiD lesion-grounded XAI Kaggle experiment
    └── scripts/      # Notebook generator

research_backlog/     # Future DR research topics
```

## Paper 1 — RETFound DR grading

Compares:

- **RETFound Baseline:** full fine-tuning + weighted cross-entropy
- **Enhanced RETFound:** LoRA + multi-scale fusion + focal/ordinal/referable losses

Primary experiments use APTOS; Messidor-2 provides external validation. Model
selection uses validation QWK.

Start with:

- `papers/01_retfound_dr_grading/notebooks/RETFound_DR_M0_vs_M2_Kaggle.ipynb`
- `papers/01_retfound_dr_grading/manuscript/arxiv_main.tex`

## Paper 2 — lesion-grounded RETFound explainability

Evaluates attention rollout, ViT Grad-CAM, and Integrated Gradients against
IDRiD lesion masks, with clinical-alignment and faithfulness statistics.

Start with:

- `papers/02_vit_retfound_explainability/planning/Plan_Explainability_ViT_RETFound.md`
- `papers/02_vit_retfound_explainability/notebooks/RETFound_IDRiD_Explainability_Journal_Kaggle.ipynb`

## Kaggle

Enable a GPU and attach the datasets/checkpoints listed in each notebook.
Kaggle outputs are written under `/kaggle/working/` and should be downloaded
into the corresponding paper topic rather than the workspace root.
