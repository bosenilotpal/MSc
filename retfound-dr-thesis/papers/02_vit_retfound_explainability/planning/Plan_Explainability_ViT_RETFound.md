# Research Plan: Explainability for ViT-RETFound (Journal Target)

**Working title:**  
*Do RETFound Explanations Align with Diabetic Retinopathy Lesions? Attention, Grad-CAM, and Concept Attribution vs Expert Annotations*

**Target type:** Full **IEEE journal** article (not workshop-only / short letter)  
**Primary venue:** *IEEE Journal of Biomedical and Health Informatics (JBHI)* — fallback *IEEE Access* / *IEEE JTEHM*; stretch *IEEE TMI*  
**Builds on:** RETFound Baseline / Enhanced RETFound checkpoints, APTOS + Messidor-2 pipeline.

**Core claim (must be supported quantitatively):**  
ViT-RETFound explanations can be evaluated for *clinical faithfulness* (overlap with lesion annotations) and *model faithfulness* (deletion/insertion), and clinically adapted fine-tuning changes what the model attends to under DR grading.

---

## 0. Journal vs MVP (scope decision)

| Scope | Content | Venue fit |
|-------|---------|-----------|
| **MVP** | Attention + Grad-CAM, one lesion dataset, Baseline vs Enhanced, qualitative panels | Conference / short letter |
| **Journal target (this plan)** | + IG/attribution, concepts, multi-dataset, error strata, faithfulness curves, failure analysis, reproducibility | Full journal |

**Rule:** Do not submit as a full journal paper until the journal checklist in §12 is complete.

---

## 1. Goal

Evaluate whether ViT-RETFound explanations for five-class DR grading are **clinically faithful**: do attention / Grad-CAM / concept-based maps highlight true lesion regions (microaneurysms, hemorrhages, exudates, neovascularization), or spurious cues (disc, edge artifacts, background)?

Secondary goal: compare **Baseline vs Enhanced** explanations under in-domain and external conditions.

---

## 2. Research questions

1. Which explanation method best overlaps with lesion annotations on fundus images?
2. Do Enhanced RETFound (LoRA + multi-scale) explanations differ from Baseline (full FT) in lesion localization?
3. Are explanations more lesion-aligned for correct predictions than for errors / adjacent-grade confusions?
4. Can concept scores (e.g., “exudate-like”, “hemorrhage-like”) predict referable vs non-referable decisions?
5. **(Journal)** Does domain shift (APTOS → Messidor-2) change explanation focus even when grade prediction is held fixed?
6. **(Journal)** Are “clinically aligned” maps also *model-faithful* (deletion/insertion), or can the two diverge?

---

## 3. Methods to compare (journal set)

| Method | Priority | Role |
|--------|----------|------|
| Attention rollout / last-layer CLS attention | Required | ViT-native baseline explanation |
| Grad-CAM / Grad-CAM++ on patch tokens | Required | Class-specific localization |
| Integrated Gradients or Attention×Grad | Required for journal | Stronger attribution fidelity |
| Concept-based (TCAV/CAV or lesion prototypes) | Required for journal | Links internals to clinical language |
| Random / center / disc-prior maps | Required control | Sanity check that methods beat trivial baselines |

**Lesion ground truth (journal needs ≥1 strong source):**
- **Primary (CONFIRMED, full five-lesion masks):**  
  https://www.kaggle.com/datasets/aaryapatel98/indian-diabetic-retinopathy-image-dataset  
  (1.01 GB, CC BY 4.0). Contains the official IDRiD layout:
  - `A. Segmentation/` — 81 fundus JPGs + per-lesion `.tif` masks  
    (`1. Microaneurysms`, `2. Haemorrhages`, `3. Hard Exudates`, `4. Soft Exudates`, `5. Optic Disc`) with train/test splits  
  - `B. Disease Grading/` — official grading CSVs (413 train / 103 test)  
  - `C. Localization/`
- **Do not use for masks:** `mohamedabdalkader/indian-diabetic-retinopathy-image-dataset-idrid` (grading + captions only; no `.tif` lesion masks; inflated row counts).  
  Also reject `realhaadkhan/idrid-segmentation-dataset` (combined masks only, ~60 MB, likely downsampled).
- **Authoritative fallback:** IEEE DataPort `A. Segmentation.zip` (~557 MB) if the Kaggle mirror disappears.
- **Secondary:** DDR or FGADR (for generalization of alignment findings)
- APTOS/Messidor: grade-level analysis only (no pixel lesions) unless you add a lesion detector proxy (disclose as weak)

**Kaggle notebook inputs (explainability run):**
1. **IDRiD (required):** `aaryapatel98/indian-diabetic-retinopathy-image-dataset`
2. RETFound CFP weights (existing)
3. Saved outputs (`M0_best.pt`, `M2_best.pt`, `splits.json`, preferably histories)
4. APTOS / Messidor-2 optional here (main-paper metrics are not retrained)

---

## 4. Experimental design (journal package)

### 4.1 Models
- RETFound Baseline (`M0_best.pt`)
- Enhanced RETFound (`M2_best.pt`)
- **Journal recommended:** A2 (best QWK) to isolate multi-scale effect on explanations
- Report grading metrics first (Acc, QWK, Ref AUROC) so XAI is not detached from performance

### 4.2 Datasets
| Role | Dataset | Use |
|------|---------|-----|
| Grade performance | APTOS test | Confirm model quality |
| External grade + explanation shift | Messidor-2 | Domain-shift XAI analysis |
| Lesion alignment (primary) | IDRiD | Quantitative overlap |
| Lesion alignment (secondary) | DDR / FGADR | Robustness of alignment conclusions |

### 4.3 Pipeline
1. Freeze checkpoint → predict grade + referable score.
2. Generate explanation map (normalize [0,1], upsample to image size).
3. Threshold heatmap (report **top-10% and top-20%**; sensitivity analysis).
4. Compare to lesion mask union **and** per-lesion type when available.
5. Stratify by: correct vs incorrect, grade, referable vs non-referable, Baseline vs Enhanced, dataset.
6. Run deletion/insertion faithfulness on the same images.

### 4.4 Quantitative metrics (all required for journal)
**Clinical alignment**
- Pointing game / hit rate
- IoU / Dice (thresholded map vs lesion mask)
- Energy pointing (heatmap mass inside lesions)
- Per-lesion breakdown (MA / hemorrhage / exudate / etc. if masks exist)

**Model faithfulness**
- Deletion AUC / Insertion AUC
- Optionally: sparseness / complexity of explanations

**Concept utility**
- Concept AUROC vs lesion presence
- Referable discrimination from concept scores

**Controls**
- Compare every method against random and optic-disc-centered priors

### 4.5 Qualitative analysis (journal expects this section)
- Multi-panel: fundus | lesion mask | attention | Grad-CAM | IG | concept overlay
- Success cases vs failure cases:
  - adjacent-grade confusion (e.g., 2↔3)
  - optic-disc / vessel / edge focus
  - correct grade but poor lesion overlap (“right for wrong reasons”)
- Optional but valuable: 2–3 clinician ratings of explanation usefulness (Likert), even small N

### 4.6 Statistics (journal expectation)
- Bootstrap CIs for overlap metrics
- Paired tests Baseline vs Enhanced on same images
- Multiple-threshold sensitivity (top-k%)
- Disclose number of images per stratum

---

## 5. Hypotheses

- H1: Grad-CAM / IG aligns better with lesions than raw CLS attention alone.
- H2: Enhanced multi-scale model focuses more on mid-level lesion cues than Baseline.
- H3: Incorrect predictions show lower lesion overlap and more disc/edge attention.
- H4: Concept scores for hemorrhage/exudate separate referable vs non-referable better than global attention alone.
- H5 **(Journal):** High clinical overlap does not always equal high faithfulness (report both).
- H6 **(Journal):** Under Messidor-2 shift, explanations become less lesion-focused even when accuracy drop is moderate.

---

## 6. Journal paper structure (recommended)

Target length: ~6,000–8,000 words + figures/tables (venue-dependent).

1. **Introduction** — trustworthy DR AI; gap: RETFound XAI rarely lesion-validated  
2. **Related work** — CNN Grad-CAM in DR; ViT explainability; foundation models; faithfulness vs plausibility  
3. **Materials and methods**
   - Datasets and lesion annotations  
   - Models (Baseline / Enhanced / A2)  
   - Explanation methods  
   - Alignment + faithfulness metrics  
   - Statistical protocol  
4. **Results**
   - Grading performance recap  
   - Method comparison tables  
   - Baseline vs Enhanced explanation differences  
   - Correct vs incorrect strata  
   - Domain-shift explanation analysis  
   - Concept results  
   - Qualitative cases  
5. **Discussion** — clinical implications, failure modes, when explanations are misleading  
6. **Limitations** — annotation incompleteness, threshold sensitivity, no full reader study (if applicable)  
7. **Conclusion**  
8. **Data/code availability** — required by many journals  

---

## 7. Required tables & figures (journal package)

### Tables
1. Dataset / annotation summary  
2. Grading performance (APTOS + Messidor-2) for models used in XAI  
3. Explanation method × alignment metrics (with CIs), Baseline vs Enhanced  
4. Alignment stratified by correct/incorrect and grade  
5. Faithfulness (deletion/insertion) by method and model  
6. Concept AUROCs (lesion presence / referable)  

### Figures
1. Method overview / pipeline schematic  
2. Multi-panel qualitative successes  
3. Multi-panel qualitative failures (disc/edge/spurious)  
4. Deletion–insertion curves  
5. Baseline vs Enhanced heatmap comparison on same eyes  
6. Optional: concept score distributions for referable DR  

---

## 8. Target IEEE venues

**Focus: IEEE journals only.**

### Primary IEEE target
- **IEEE Journal of Biomedical and Health Informatics (JBHI)** — best fit for clinically grounded ML + explainability; strong reputation, reasonable scope match.

### IEEE fallbacks (faster / broader)
- **IEEE Access** — broad, faster review, open access; good if JBHI rejects on novelty. Accepts applied XAI if rigorous.
- **IEEE Journal of Translational Engineering in Health and Medicine (JTEHM)** — open access, translational clinical focus.
- **IEEE Open Journal of Engineering in Medicine and Biology (OJEMB)** — open access, shorter format acceptable.

### IEEE stretch (needs multi-dataset + concepts + clinician component)
- **IEEE Transactions on Medical Imaging (TMI)** — high bar; requires strong methodological novelty (lesion-grounded ViT faithfulness framework, not just applying Grad-CAM).
- **IEEE Transactions on Biomedical Engineering (TBME)** — possible if framed around methodology + validation.

### Preprint / submission policy
- arXiv (`eess.IV` primary, `cs.CV` secondary) is compatible with IEEE (IEEE permits preprints; add the required IEEE copyright notice once accepted).
- Do **not** post the final IEEE-typeset PDF; post accepted author version per IEEE policy.

**Practical recommendation:** target **JBHI first**. If desk-rejected for scope/novelty, move to **IEEE Access** (rigor over novelty). Reserve **TMI** only if IDRiD+DDR, IG+concepts, and a small clinician review are all complete.

### IEEE formatting / submission requirements (prepare early)
- Use the **IEEE journal LaTeX template** (`IEEEtran`, `journal` option) — convert `arxiv_main.tex` accordingly.
- **Bibliography:** IEEE style (`IEEEtran.bst`), numbered `\cite{}`.
- **Figures:** ≥300 dpi; vector (PDF/EPS) preferred for plots; grayscale-legible.
- **Graphical abstract** + short author bios/photos (JBHI/Access).
- **ORCID** for all authors.
- **Ethics/data statement:** all datasets public (APTOS, Messidor-2, IDRiD) — state no new patient data / IRB not required, cite dataset licenses.
- **Reproducibility:** code + config availability statement (IEEE encourages).
- Check page/overlength charges (esp. IEEE Access APC; JBHI overlength fees).

---

## 9. Implementation plan (journal timeline)

| Phase | Tasks | Est. effort |
|-------|-------|-------------|
| **P0** | IDRiD (+DDR if possible); heatmap export utils | 3–4 days |
| **P1** | Attention rollout + Grad-CAM | 3–4 days |
| **P2** | IG / Attention×Grad + random/disc controls | 3–4 days |
| **P3** | Alignment metrics, CIs, strata tables | 3–4 days |
| **P4** | Faithfulness deletion/insertion | 2–3 days |
| **P5** | Baseline vs Enhanced vs A2 explanation comparison | 2–3 days |
| **P6** | Messidor-2 explanation-shift analysis | 2–3 days |
| **P7** | Concept/CAV or lesion-prototype probes | 4–6 days |
| **P8** | Qualitative figures + optional clinician Likert | 3–5 days |
| **P9** | Full manuscript + reproducibility package | 1–2 weeks |

**Total:** ~6–9 weeks focused work after lesion-data access (journal package).  
MVP can still be cut at end of P3–P4 for a short paper if needed.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| No pixel lesions on APTOS | Alignment on IDRiD/DDR; APTOS for grade/ROC only |
| ViT Grad-CAM noisy | Report attention + IG; multi-threshold sensitivity |
| Plausible but unfaithful maps | Always pair overlap with deletion/insertion |
| Scope creep | Lock “journal required” list in §12; park extras |
| Reviewer: “not novel, just Grad-CAM” | Emphasize foundation-model + lesion-grounded + Baseline vs Enhanced clinical adaptation |
| Weak clinician validation | At least expert qualitative review of failure cases |

---

## 11. What is *not* enough for a journal

- Grad-CAM montages without lesion overlap metrics  
- Single dataset, single method, no Baseline vs Enhanced  
- No faithfulness evaluation  
- No failure-case analysis  
- Framing as “we applied XAI” without a clinical research question  

---

## 12. Journal submission checklist

Submit only when all **Required** items are done:

### Required
- [ ] ≥2 explanation methods beyond cherry-picked visuals (attention + Grad-CAM minimum; IG strongly preferred)
- [ ] Lesion-mask quantitative alignment on IDRiD (or equivalent)
- [ ] Control baselines (random / disc prior)
- [ ] Baseline vs Enhanced comparison on identical images
- [ ] Correct vs incorrect (or referable) strata
- [ ] Deletion and/or insertion faithfulness
- [ ] Threshold sensitivity (at least two top-k settings)
- [ ] Qualitative success **and** failure panels
- [ ] Limitations + reproducibility (code/config/checkpoint note)
- [ ] Preprint/journal policy checked

### Strongly recommended (raises acceptance odds)
- [ ] Second lesion dataset (DDR/FGADR)
- [ ] Concept-based explanations
- [ ] A2 ablation for multi-scale explanation effect
- [ ] Messidor-2 explanation-shift analysis
- [ ] Bootstrap CIs + paired statistical tests
- [ ] Small clinician usefulness rating

### IEEE-specific (required before IEEE submission)
- [ ] Manuscript in `IEEEtran` journal template
- [ ] IEEE numbered citations + `IEEEtran.bst`
- [ ] Figures ≥300 dpi, vector plots where possible
- [ ] ORCID for all authors
- [ ] Data/ethics statement (public datasets; licenses cited)
- [ ] Code/reproducibility availability statement
- [ ] Graphical abstract (JBHI/Access)
- [ ] Author bios (+ photos if required by venue)
- [ ] Overlength/APC cost checked for chosen IEEE journal

### Optional / stretch
- [ ] Full reader study (diagnosis with/without explanations)
- [ ] Prospective clinical data

---

## 13. Immediate next actions

1. ~~Confirm IDRiD access~~ → **Done**. Use **`aaryapatel98/indian-diabetic-retinopathy-image-dataset`** (not the mohamedabdalkader grading-only mirror).  
2. On Kaggle: **Add Input** → attach that dataset + RETFound CFP weights + saved `M0_best.pt` / `M2_best.pt`.  
3. Re-run the explainability notebook inventory cell; expect lesion-mask counts near MA=81, EX=81, HE=80, SE=40, OD=81 and official grading labels = 516.  
4. Smoke-test heatmaps on ~50 lesion-positive images, then raise `XAI_MAX_IMAGES` for the full journal run.  
5. Mask union policy: lesion union = MA ∪ HE ∪ EX ∪ SE (**exclude** optic disc from “lesion” GT; use OD as a *spurious-focus* control).  
6. Lock venue shortlist (JBHI first; IEEE Access fallback).  
7. Freeze journal figure/table list (§7) before full runs.

---

## 14. Success criteria for “journal-ready”

The manuscript can claim journal readiness when it can answer, with tables:

> *Which RETFound explanations are lesion-aligned, which are merely plausible, how clinical adaptation changes attention, and when explanations fail under domain shift?*
