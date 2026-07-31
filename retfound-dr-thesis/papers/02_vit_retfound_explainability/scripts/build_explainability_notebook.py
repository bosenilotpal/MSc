"""Generate the self-contained Kaggle explainability experiment notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "RETFound_IDRiD_Explainability_Journal_Kaggle.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


cells = [
md(r"""# Lesion-Grounded Explainability for ViT-RETFound (Journal Experiment)

This notebook evaluates whether **RETFound Baseline** and **Enhanced RETFound** explanations align with diabetic-retinopathy lesions in **IDRiD**.

### Kaggle inputs to attach
1. **IDRiD (required, full masks):** `aaryapatel98/indian-diabetic-retinopathy-image-dataset`  
   Must include `A. Segmentation` (per-lesion `.tif` masks for MA/HE/EX/SE/OD) and preferably `B. Disease Grading` (official CSVs).  
   Do **not** use `mohamedabdalkader/indian-diabetic-retinopathy-image-dataset-idrid` — grading/captions only, no lesion masks.
2. RETFound CFP weights (`RETFound_cfp_weights.pth`)
3. Saved outputs containing `M0_best.pt`, `M2_best.pt`, `splits.json`, and preferably `M0_history.csv` / `M2_history.csv`
4. APTOS and Messidor-2 are optional here (main-paper metrics are not retrained)

### Outputs
- Dataset inventory and grading metrics (accuracy, macro-F1, QWK, referable AUROC)
- Confusion matrices and referable ROC curves
- Existing training loss/QWK curves
- Attention rollout, ViT Grad-CAM, and Integrated Gradients heatmaps
- Lesion overlap: pointing game, energy-in-lesion, IoU, Dice at top 10%/20%
- Deletion/insertion faithfulness curves
- Bootstrap 95% CIs and paired Wilcoxon tests
- Paper-ready PNG/PDF figures, CSV tables, JSON summaries, and ZIP archive

> **Important:** This notebook evaluates frozen checkpoints; it does not retrain the main models. Defaults are a journal-quality pilot that fits a Kaggle T4. Increase sample limits only after the smoke test succeeds.
"""),

code(r"""# ========================= INSTALL + IMPORTS =========================
!pip -q install timm==1.0.15

from pathlib import Path
from dataclasses import dataclass
import os, re, json, math, random, shutil, warnings, zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, confusion_matrix,
    roc_auc_score, roc_curve, auc
)
from scipy.stats import wilcoxon, spearmanr, binomtest
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="paper")

@dataclass
class CFG:
    IS_KAGGLE = Path("/kaggle/input").exists()
    INPUT_ROOT = Path("/kaggle/input") if IS_KAGGLE else Path("./data")
    OUT_DIR = Path("/kaggle/working/retfound_xai_outputs") if IS_KAGGLE else Path("./retfound_xai_outputs")
    FIG_DIR = OUT_DIR / "figures"
    TABLE_DIR = OUT_DIR / "tables"
    HEATMAP_DIR = OUT_DIR / "heatmaps"

    IMG_SIZE = 224
    NUM_CLASSES = 5
    SEED = 42
    NUM_WORKERS = 2
    BATCH_SIZE = 16

    MS_BLOCKS = (7, 15, 23)
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGETS = ("qkv",)

    # Start with 50. Increase to all lesion-positive images only after the smoke test.
    XAI_MAX_IMAGES = 50
    IG_MAX_IMAGES = 25
    FAITHFULNESS_MAX_IMAGES = 25
    IG_STEPS = 16
    PERTURB_STEPS = 20
    TOP_K = (0.10, 0.20)
    BOOTSTRAP_N = 1000

    # Set True only after the first complete run.
    RUN_INTEGRATED_GRADIENTS = True
    RUN_FAITHFULNESS = True

for d in (CFG.OUT_DIR, CFG.FIG_DIR, CFG.TABLE_DIR, CFG.HEATMAP_DIR):
    d.mkdir(parents=True, exist_ok=True)

def seed_everything(seed=CFG.SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
print("Output:", CFG.OUT_DIR)
"""),

code(r"""# ========================= DISCOVER INPUTS + INVENTORY IDRiD =========================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def all_files(root, exts=None):
    root = Path(root)
    if not root.exists():
        return []
    xs = [p for p in root.rglob("*") if p.is_file()]
    return xs if exts is None else [p for p in xs if p.suffix.lower() in exts]

# Manual overrides: paste a path here if auto-detection picks the wrong dataset.
IDRID_ROOT_OVERRIDE = ""
WEIGHTS_PATH_OVERRIDE = ""
SAVED_OUT_OVERRIDE = ""

LESION_SUFFIX_RE = re.compile(r"(?i)[_\-\s](MA|HE|EX|SE|OD)\.")
LESION_DIR_KEYS = ("microaneurysm", "haemorrhage", "hemorrhage", "exudate", "optic disc", "optic disk")
OTHER_DATASET_KEYS = ("messidor", "aptos", "eyepacs", "ddr", "deepdrid", "kaggle-dr")

def mask_like_count(root):
    # Files that look like lesion ground truth: coded name suffix or lesion-named folder.
    n = 0
    for p in all_files(root, IMG_EXTS):
        if LESION_SUFFIX_RE.search(p.name) or any(k in str(p).lower() for k in LESION_DIR_KEYS):
            n += 1
    return n

def candidate_dirs(max_depth=3):
    # Inputs are sometimes mounted nested (e.g. /kaggle/input/datasets/<owner>/<slug>),
    # so enumerate a few levels deep rather than only the top level.
    out, stack = [], [(CFG.INPUT_ROOT, 0)]
    while stack:
        d, depth = stack.pop()
        if depth >= max_depth:
            continue
        try:
            children = sorted(d.glob("*"))
        except OSError:
            continue
        for p in children:
            if p.is_dir():
                out.append(p)
                stack.append((p, depth + 1))
    return out

def print_tree(root, max_depth=3, max_per_dir=12):
    root = Path(root)
    print(f"\nTree of {root} (depth {max_depth}):")
    def walk(d, depth, prefix):
        if depth > max_depth:
            return
        try:
            entries = sorted(d.glob("*"))
        except OSError:
            return
        dirs = [p for p in entries if p.is_dir()]
        files = [p for p in entries if p.is_file()]
        for p in dirs[:max_per_dir]:
            n_img = len(all_files(p, IMG_EXTS))
            print(f"{prefix}{p.name}/  [{n_img} images]")
            walk(p, depth + 1, prefix + "  ")
        if len(dirs) > max_per_dir:
            print(f"{prefix}... (+{len(dirs)-max_per_dir} more dirs)")
        for p in files[:4]:
            print(f"{prefix}{p.name}")
        if len(files) > 4:
            print(f"{prefix}... (+{len(files)-4} more files)")
    walk(root, 1, "  ")

def describe_inputs():
    # Print every attached input so a mismatched dataset name is immediately visible.
    print("INPUT_ROOT:", CFG.INPUT_ROOT, "| exists:", CFG.INPUT_ROOT.exists())
    if not CFG.INPUT_ROOT.exists():
        print("No input directory. On Kaggle use 'Add Input'; locally create ./data.")
        return []
    dirs = candidate_dirs()
    if not dirs:
        print("INPUT_ROOT is empty - nothing attached yet.")
    rows = []
    for d in dirs:
        imgs = all_files(d, IMG_EXTS)
        if not imgs and not any(p.suffix.lower() in {".pt", ".pth", ".csv"} for p in all_files(d)):
            continue
        rows.append({
            "dir": str(d.relative_to(CFG.INPUT_ROOT)),
            "depth": len(d.relative_to(CFG.INPUT_ROOT).parts),
            "images": len(imgs),
            "mask_like": mask_like_count(d),
            "checkpoints": len([p for p in all_files(d) if p.suffix.lower() in {".pt", ".pth"}]),
            "csv_files": len([p for p in all_files(d) if p.suffix.lower() == ".csv"]),
        })
    if rows:
        display(pd.DataFrame(rows).sort_values(["depth", "dir"]).head(40))
    return dirs

ATTACHED_INPUTS = describe_inputs()

def _fail(what, hint):
    names = ", ".join(str(p.relative_to(CFG.INPUT_ROOT)) for p in ATTACHED_INPUTS[:20]) or "<none>"
    raise FileNotFoundError(
        f"{what} not found.\nCandidate input dirs: {names}\n{hint}\n"
        "If the dataset is attached under an unexpected name, set the matching "
        "*_OVERRIDE variable at the top of this cell to its /kaggle/input path."
    )

PREFERRED_IDRID_SLUGS = (
    "aaryapatel98",  # audited: full A. Segmentation + B. Disease Grading
    "indian-diabetic-retinopathy-image-dataset",  # same slug without owner prefix
)

def resolve_idrid_root():
    if IDRID_ROOT_OVERRIDE:
        return Path(IDRID_ROOT_OVERRIDE)
    # Prefer mask-rich IDRiD roots. If both aaryapatel98 (full masks) and a
    # grading-only mirror are attached, the mask score must win.
    scored = []
    for d in ATTACHED_INPUTS:
        if not d.is_dir() or any(k in str(d).lower() for k in OTHER_DATASET_KEYS):
            continue
        imgs = all_files(d, IMG_EXTS)
        if not imgs:
            continue
        path_l = str(d).lower()
        name_l = d.name.lower()
        is_idrid = "idrid" in name_l or "idrid" in path_l or "indian-diabetic-retinopathy" in path_l
        if not is_idrid:
            continue
        n_masks = mask_like_count(d)
        preferred = any(s in path_l for s in PREFERRED_IDRID_SLUGS)
        # Prefer shallower dataset roots over deep lesion subfolders, but keep
        # mask count as the dominant signal so grading-only mirrors lose.
        depth_penalty = len(d.relative_to(CFG.INPUT_ROOT).parts)
        score = (10_000_000 if n_masks > 0 else 0) + (1_000_000 if preferred else 0) + 100 * n_masks + len(imgs) - depth_penalty
        scored.append((score, n_masks, len(imgs), d))
    if not scored:
        _fail(
            "IDRiD input",
            "Attach: aaryapatel98/indian-diabetic-retinopathy-image-dataset "
            "(must include A. Segmentation with per-lesion .tif masks)."
        )
    scored.sort(key=lambda t: t[0], reverse=True)
    _, n_masks, n_imgs, best = scored[0]
    print(f"Selected IDRiD root: {best}\n  images={n_imgs}, mask-like files={n_masks}")
    if n_masks == 0:
        print(
            "  WARNING: no lesion-mask files detected. Detach grading-only mirrors "
            "(e.g. mohamedabdalkader/...) and attach aaryapatel98/indian-diabetic-retinopathy-image-dataset."
        )
    return best

def resolve_weights():
    if WEIGHTS_PATH_OVERRIDE:
        return str(WEIGHTS_PATH_OVERRIDE)
    def rank(p):
        n = p.name.lower()
        if "retfound" in n and "cfp" in n: return 0
        if "retfound" in n: return 1
        if n in {"m0_best.pt", "m2_best.pt"}: return 9  # fine-tuned runs, not pretrained weights
        return 5
    weights = [p for p in all_files(CFG.INPUT_ROOT) if p.suffix.lower() in {".pth", ".pt"}]
    weights = [p for p in sorted(weights, key=rank) if rank(p) < 9]
    if not weights:
        _fail("RETFound CFP weights", "Attach a dataset containing RETFound_cfp_weights.pth")
    return str(weights[0])

def resolve_saved_out():
    if SAVED_OUT_OVERRIDE:
        return Path(SAVED_OUT_OVERRIDE)
    m0 = [p for p in CFG.INPUT_ROOT.rglob("M0_best.pt") if p.is_file()]
    for p in m0:
        if (p.parent / "M2_best.pt").is_file():
            return p.parent
    if m0:
        print("WARNING: found M0_best.pt without M2_best.pt in", m0[0].parent)
        return m0[0].parent
    _fail("Saved checkpoints (M0_best.pt / M2_best.pt)",
          "Attach the Kaggle dataset created from your earlier /kaggle/working/outputs run.")

IDRID_ROOT = resolve_idrid_root()
WEIGHTS_PATH = resolve_weights()
SAVED_OUT = resolve_saved_out()
print("\nIDRiD:", IDRID_ROOT)
print("RETFound weights:", WEIGHTS_PATH)
print("Saved outputs:", SAVED_OUT)
print_tree(IDRID_ROOT, max_depth=3)

for name in ["M0_best.pt", "M2_best.pt", "M0_history.csv", "M2_history.csv", "splits.json"]:
    src = SAVED_OUT / name
    if src.exists():
        shutil.copy2(src, CFG.OUT_DIR / name)

files = all_files(IDRID_ROOT)
inventory = pd.DataFrame([{
    "path": str(p.relative_to(IDRID_ROOT)),
    "suffix": p.suffix.lower(),
    "size_bytes": p.stat().st_size,
} for p in files])
inventory.to_csv(CFG.TABLE_DIR / "idrid_file_inventory.csv", index=False)
print("\nTop extensions:")
display(inventory["suffix"].value_counts().head(15).to_frame("count"))
print("\nCSV files:")
for p in [x for x in files if x.suffix.lower() == ".csv"]:
    print(" -", p.relative_to(IDRID_ROOT))
print("\nImage-like files:", len([x for x in files if x.suffix.lower() in IMG_EXTS]))
"""),

code(r"""# ========================= BUILD IDRiD IMAGE/MASK/LABEL INDEX =========================
LESION_CODES = {
    "MA": ["microaneurysm", "microaneurysms"],
    "HE": ["haemorrhage", "haemorrhages", "hemorrhage", "hemorrhages"],
    "EX": ["hard exudate", "hard exudates"],
    "SE": ["soft exudate", "soft exudates"],
    "OD": ["optic disc", "optic disk"],
}

def infer_mask_type(path):
    s = str(path).lower().replace("_", " ")
    stem = path.stem.upper()
    for code, names in LESION_CODES.items():
        if re.search(rf"(?:^|[_\-\s]){code}(?:$|[_\-\s])", stem):
            return code
        if any(name in s for name in names):
            return code
    return None

def infer_split(path_or_name):
    s = str(path_or_name).lower()
    if "testing" in s or "test set" in s or re.search(r"(?:^|[_\-/\s])test(?:[_\-/\s]|$)", s):
        return "TEST"
    if "training" in s or "train set" in s or re.search(r"(?:^|[_\-/\s])train(?:[_\-/\s]|$)", s):
        return "TRAIN"
    return "UNK"

def infer_task(path):
    # A. Segmentation and B. Disease Grading reuse IDRiD_NNN names independently;
    # never merge them by stem alone.
    s = str(path).lower()
    if "segmentation" in s:
        return "SEG"
    if "grading" in s or "disease" in s:
        return "GRADE"
    return "OTHER"

def idrid_number(stem):
    stem = re.sub(r"(?i)[_\-\s](MA|HE|EX|SE|OD)$", "", Path(stem).stem)
    m = re.search(r"(?i)IDRiD[_\-\s]*(\d+)", stem)
    return int(m.group(1)) if m else None

def canonical_id(path, split=None, task=None):
    # Keys look like GRADE_TRAIN_IDRID_001 / SEG_TEST_IDRID_012 so train/test
    # disease-grading images (both named IDRiD_001.jpg) stay distinct, and
    # segmentation IDRiD_01 is never collapsed into grading IDRiD_001.
    path = Path(path)
    num = idrid_number(path.stem)
    base = f"IDRID_{num:03d}" if num is not None else path.stem.upper()
    task = task or infer_task(path)
    split = split or infer_split(path)
    if task == "OTHER" and split == "UNK":
        return base
    return f"{task}_{split}_{base}"

image_files = all_files(IDRID_ROOT, IMG_EXTS)
mask_files = [p for p in image_files if infer_mask_type(p) is not None]
original_files = [p for p in image_files if infer_mask_type(p) is None]

# Exclude obvious derived/visualization folders from originals.
original_files = [p for p in original_files if not any(k in str(p).lower() for k in ["mask", "groundtruth", "ground truth"])]

image_map = {}
for p in original_files:
    cid = canonical_id(p)
    # Prefer paths explicitly containing "original" and JPEG over ambiguous duplicates.
    score = (10 if "original" in str(p).lower() else 0) + (2 if p.suffix.lower() in {".jpg", ".jpeg"} else 0)
    if cid not in image_map or score > image_map[cid][0]:
        image_map[cid] = (score, p)

mask_map = defaultdict(dict)
for p in mask_files:
    # Masks live under A. Segmentation; attach them to the matching SEG image id.
    cid = canonical_id(p, task="SEG")
    mask_map[cid][infer_mask_type(p)] = p

# Grading labels: require the OFFICIAL IDRiD CSVs (413 train + 103 test = 516 rows).
# Community mirrors add caption/class columns and inflate row counts, which would make
# reported class distributions irreproducible against the published dataset.
OFFICIAL_GRADE_COLS = {"image name", "retinopathy grade", "risk of macular edema"}
NON_OFFICIAL_COLS = {"class", "caption", "text", "prompt", "description"}
IDRID_OFFICIAL_COUNTS = {"train": 413, "test": 103}

# Search all inputs (not just IDRID_ROOT) so grading CSVs may live in a separate mirror.
candidate_csvs = [p for p in all_files(CFG.INPUT_ROOT)
                  if p.suffix.lower() == ".csv"
                  and not any(k in str(p).lower() for k in OTHER_DATASET_KEYS)]

def inspect_grading_csv(p):
    try:
        cols = [str(c).strip().lower() for c in pd.read_csv(p, nrows=5).columns]
    except Exception:
        return None
    if not any("retinopathy" in c or c == "grade" for c in cols):
        return None
    extras = sorted(set(cols) & NON_OFFICIAL_COLS)
    return {
        "path": p,
        "cols": cols,
        "official_schema": OFFICIAL_GRADE_COLS.issubset(set(cols)) and not extras,
        "extra_cols": extras,
        "split": "train" if "train" in p.name.lower() else ("test" if "test" in p.name.lower() else "?"),
    }

found = [m for m in (inspect_grading_csv(p) for p in candidate_csvs) if m]
official = [m for m in found if m["official_schema"]]
GRADE_LABEL_SOURCE = "official" if official else "non-official-fallback"

if not official and found:
    print("WARNING: no official IDRiD grading CSV found. Rejected candidates:")
    for m in found:
        print(f"  - {m['path']} | extra columns: {m['extra_cols'] or 'none'}")
    print("  Falling back to these labels; class distributions will NOT match published IDRiD.")
    print("  For the paper, attach 'B. Disease Grading' with")
    print("  IDRiD_Disease_Grading_Training_Labels.csv / _Testing_Labels.csv.")

use_csvs = official if official else found
labels, label_rows = {}, {}
for m in use_csvs:
    # Official training CSV often has trailing spaces / Unnamed Excel columns; keep the three real fields.
    g = pd.read_csv(m["path"])
    g.columns = [str(c).strip() for c in g.columns]
    keep = [c for c in g.columns if not re.match(r"(?i)^unnamed", c)]
    g = g[keep]
    label_rows[str(m["path"])] = len(g)
    print(f"Grading CSV ({'OFFICIAL' if m['official_schema'] else 'non-official'}): {m['path']}")
    print("  Columns:", list(g.columns), "| rows:", len(g))
    exp = IDRID_OFFICIAL_COUNTS.get(m["split"])
    if exp and len(g) != exp:
        print(f"  WARNING: expected {exp} rows for the {m['split']} split, found {len(g)}.")
    id_col = next((c for c in g.columns if "image" in c.lower() or c.lower() in {"id", "image_id"}), g.columns[0])
    grade_col = next((c for c in g.columns if "retinopathy" in c.lower() or "grade" in c.lower()), None)
    split = {"train": "TRAIN", "test": "TEST"}.get(m["split"], infer_split(m["path"]))
    if grade_col:
        for _, row in g.iterrows():
            try:
                cid = canonical_id(Path(str(row[id_col])), split=split, task="GRADE")
                labels[cid] = int(row[grade_col])
            except Exception:
                pass

print(f"\nGrade label source: {GRADE_LABEL_SOURCE} | unique labelled ids: {len(labels)}")
if GRADE_LABEL_SOURCE == "official" and len(labels) != sum(IDRID_OFFICIAL_COUNTS.values()):
    print(f"WARNING: {len(labels)} labelled ids but official IDRiD grading has "
          f"{sum(IDRID_OFFICIAL_COUNTS.values())}. Check that both split CSVs are attached.")
else:
    print(f"  train labels: {sum(1 for k in labels if '_TRAIN_' in k)} | "
          f"test labels: {sum(1 for k in labels if '_TEST_' in k)}")

# Persist provenance so the paper's data section can be verified later.
(CFG.OUT_DIR / "label_provenance.json").write_text(json.dumps({
    "grade_label_source": GRADE_LABEL_SOURCE,
    "csvs_used": {str(m["path"]): label_rows.get(str(m["path"])) for m in use_csvs},
    "csvs_rejected": [str(m["path"]) for m in found if m not in use_csvs],
    "n_labelled_ids": len(labels),
}, indent=2))

rows = []
for cid in sorted(set(image_map) | set(mask_map)):
    if cid not in image_map:
        continue
    row = {
        "image_id": cid,
        "image_path": str(image_map[cid][1]),
        "task": cid.split("_")[0] if "_" in cid else "OTHER",
        "split": cid.split("_")[1] if cid.count("_") >= 2 else "UNK",
        "label": labels.get(cid, np.nan),
    }
    for code in LESION_CODES:
        row[f"mask_{code}"] = str(mask_map.get(cid, {}).get(code, ""))
    row["n_lesion_masks"] = sum(bool(row[f"mask_{c}"]) for c in ("MA", "HE", "EX", "SE"))
    rows.append(row)

idrid_df = pd.DataFrame(rows)
if len(idrid_df) == 0:
    raise RuntimeError("No IDRiD originals were matched. Inspect idrid_file_inventory.csv and adjust path rules.")
idrid_df.to_csv(CFG.TABLE_DIR / "idrid_index.csv", index=False)

n_grade = int((idrid_df.task == "GRADE").sum()) if "task" in idrid_df else 0
n_seg = int((idrid_df.task == "SEG").sum()) if "task" in idrid_df else 0
print("Original images:", len(idrid_df), f"(GRADE={n_grade}, SEG={n_seg})")
print("With lesion masks:", int((idrid_df.n_lesion_masks > 0).sum()))
print("With grade label:", int(idrid_df.label.notna().sum()),
      f"(of {n_grade} disease-grading images)")
print("\nSample disease-grading rows:")
display(idrid_df[idrid_df.task == "GRADE"].head())
print("\nSample segmentation (lesion-mask) rows:")
display(idrid_df[idrid_df.n_lesion_masks > 0].head())
display(pd.DataFrame({
    "mask_type": list(LESION_CODES),
    "available_images": [int(idrid_df[f"mask_{c}"].astype(bool).sum()) for c in LESION_CODES]
}))

if int((idrid_df.n_lesion_masks > 0).sum()) == 0:
    raise RuntimeError(
        "No lesion masks found, so the lesion-alignment study cannot run.\n"
        f"Resolved IDRiD root: {IDRID_ROOT}\n"
        "Attach this Kaggle dataset (verified complete five-lesion masks):\n"
        "  aaryapatel98/indian-diabetic-retinopathy-image-dataset\n"
        "Expected layout: A. Segmentation/2. All Segmentation Groundtruths/"
        "{a. Training Set,b. Testing Set}/"
        "{1. Microaneurysms,2. Haemorrhages,3. Hard Exudates,4. Soft Exudates,5. Optic Disc}/\n"
        "Do NOT use mohamedabdalkader/...-idrid (grades/captions only) or "
        "realhaadkhan/idrid-segmentation-dataset (combined masks, downsampled)."
    )

# Journal XAI subset: lesion-positive images, deterministic sample.
xai_df = idrid_df[idrid_df.n_lesion_masks > 0].copy()
if len(xai_df) > CFG.XAI_MAX_IMAGES:
    xai_df = xai_df.sample(CFG.XAI_MAX_IMAGES, random_state=CFG.SEED)
xai_df = xai_df.sort_values("image_id").reset_index(drop=True)
print("XAI subset:", len(xai_df))
print("Note: SEG images use a separate IDRiD numbering from B. Disease Grading,")
print("      so grade-stratified XAI is only available when a SEG row has a matched label.")"""),

code(r"""# ========================= MODEL DEFINITIONS + CHECKPOINT LOAD =========================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_vit_backbone(num_classes=0, weights_path=""):
    model = timm.create_model(
        "vit_large_patch16_224", pretrained=(weights_path == ""),
        num_classes=num_classes, global_pool="token"
    )
    if weights_path:
        try: ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        except TypeError: ckpt = torch.load(weights_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        cleaned, model_sd = {}, model.state_dict()
        for k, v in state.items():
            nk = k.replace("module.", "")
            if nk.startswith("head") or nk.startswith("fc_norm"): continue
            if nk in model_sd and model_sd[nk].shape == v.shape:
                cleaned[nk] = v
        model.load_state_dict(cleaned, strict=False)
        print(f"Loaded RETFound CFP weights | matched={len(cleaned)}")
    return model

class M0RetFound(nn.Module):
    def __init__(self, num_classes=5, weights_path=""):
        super().__init__()
        self.backbone = load_vit_backbone(num_classes, weights_path)
    def forward(self, x):
        return self.backbone(x)

class MultiScaleFusionHead(nn.Module):
    def __init__(self, dim=1024, num_classes=5, n_scales=3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim*n_scales, dim), nn.GELU(), nn.Dropout(0.1))
        self.classifier = nn.Linear(dim, num_classes)
        self.ordinal = nn.Linear(dim, num_classes-1)
        self.referable = nn.Linear(dim, 1)
    def forward(self, feats):
        h = self.proj(torch.cat(feats, dim=-1))
        return {
            "logits": self.classifier(h),
            "ordinal": self.ordinal(h),
            "referable": self.referable(h).squeeze(-1),
        }

class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

def inject_lora_timm_vit(model, r=8, alpha=16, dropout=0.05, target_names=("qkv",)):
    n = 0
    for name, module in list(model.named_modules()):
        if name.split(".")[-1] not in target_names or not isinstance(module, nn.Linear):
            continue
        parent_path = name.rsplit(".", 1)
        parent = model if len(parent_path) == 1 else model.get_submodule(parent_path[0])
        child = name if len(parent_path) == 1 else parent_path[1]
        setattr(parent, child, LoRALinear(module, r, alpha, dropout)); n += 1
    print("LoRA layers injected:", n)
    return n

class M2RetFound(nn.Module):
    def __init__(self, num_classes=5, weights_path="", ms_blocks=(7,15,23),
                 lora_r=8, lora_alpha=16, lora_dropout=0.05, lora_targets=("qkv",)):
        super().__init__()
        self.ms_blocks = tuple(ms_blocks)
        self.backbone = load_vit_backbone(0, weights_path)
        for p in self.backbone.parameters(): p.requires_grad = False
        inject_lora_timm_vit(self.backbone, lora_r, lora_alpha, lora_dropout, lora_targets)
        dim = getattr(self.backbone, "embed_dim", 1024)
        self.head = MultiScaleFusionHead(dim, num_classes, len(self.ms_blocks))
        self._hooks, self._cache = [], {}
        self._register_hooks()
    def _register_hooks(self):
        def make_hook(i):
            def hook(_m, _i, out):
                self._cache[i] = 0.5 * (out[:,0] + out[:,1:].mean(dim=1))
            return hook
        for i in self.ms_blocks:
            self._hooks.append(self.backbone.blocks[i].register_forward_hook(make_hook(i)))
    def forward(self, x):
        self._cache = {}
        _ = self.backbone.forward_features(x)
        return self.head([self._cache[i] for i in self.ms_blocks])

def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(Path(path).name, "| epoch:", ckpt.get("epoch"), "| val:", ckpt.get("val"))
    print("missing:", len(missing), "unexpected:", len(unexpected))
    model.eval()
    return model

baseline_model = load_checkpoint(
    M0RetFound(CFG.NUM_CLASSES, WEIGHTS_PATH).to(DEVICE),
    CFG.OUT_DIR / "M0_best.pt"
)
enhanced_model = load_checkpoint(
    M2RetFound(CFG.NUM_CLASSES, WEIGHTS_PATH, CFG.MS_BLOCKS,
               CFG.LORA_R, CFG.LORA_ALPHA, CFG.LORA_DROPOUT, CFG.LORA_TARGETS).to(DEVICE),
    CFG.OUT_DIR / "M2_best.pt"
)
MODELS = {"Baseline": baseline_model, "Enhanced": enhanced_model}
for name, model in MODELS.items():
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name} adaptation parameters: trainable={trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
    # XAI is inference-only. Freeze parameters to avoid allocating ~GB of parameter gradients;
    # gradients still flow to the input / retained token activations.
    for p in model.parameters(): p.requires_grad = False
    model.eval()
print("Models loaded.")
"""),

code(r"""# ========================= DATASET + GRADING EVALUATION =========================
def preprocess_pil(img):
    img = img.convert("RGB").resize((CFG.IMG_SIZE, CFG.IMG_SIZE), Image.BICUBIC)
    rgb = np.asarray(img).astype(np.float32) / 255.0
    x = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2,0,1).float(), rgb

def read_binary_mask(path):
    if not path or not Path(path).is_file():
        return np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE), dtype=np.uint8)
    m = Image.open(path).convert("L").resize((CFG.IMG_SIZE, CFG.IMG_SIZE), Image.NEAREST)
    return (np.asarray(m) > 0).astype(np.uint8)

class IDRiDDataset(Dataset):
    def __init__(self, frame, return_masks=False):
        self.df = frame.reset_index(drop=True)
        self.return_masks = return_masks
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x, rgb = preprocess_pil(Image.open(row.image_path))
        y = int(row.label) if pd.notna(row.label) else -1
        if not self.return_masks:
            return x, y, row.image_id
        masks = {c: read_binary_mask(row[f"mask_{c}"]) for c in LESION_CODES}
        lesion = np.logical_or.reduce([masks[c] > 0 for c in ("MA","HE","EX","SE")]).astype(np.uint8)
        return x, y, row.image_id, rgb, masks, lesion

def logits_of(model, x):
    out = model(x)
    return out["logits"] if isinstance(out, dict) else out

@torch.no_grad()
def predict_frame(model, frame):
    ds = IDRiDDataset(frame, False)
    dl = DataLoader(ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                    num_workers=CFG.NUM_WORKERS, pin_memory=True)
    ys, ids, probs = [], [], []
    for x, y, image_id in dl:
        p = torch.softmax(logits_of(model, x.to(DEVICE)), dim=1).cpu().numpy()
        ys.extend(y.numpy().tolist()); ids.extend(list(image_id)); probs.append(p)
    probs = np.concatenate(probs)
    pred = probs.argmax(1)
    return pd.DataFrame({
        "image_id": ids, "y_true": ys, "y_pred": pred,
        "referable_score": probs[:,2:].sum(1),
        **{f"p{i}": probs[:,i] for i in range(CFG.NUM_CLASSES)}
    })

def grading_metrics(pred_df):
    d = pred_df[pred_df.y_true >= 0]
    if len(d) == 0: return {}
    y, p = d.y_true.values, d.y_pred.values
    out = {
        "n": len(d),
        "accuracy": accuracy_score(y,p),
        "macro_f1": f1_score(y,p,average="macro"),
        "qwk": cohen_kappa_score(y,p,weights="quadratic"),
        "referable_accuracy": accuracy_score(y>=2,p>=2),
    }
    try: out["referable_auroc"] = roc_auc_score(y>=2,d.referable_score)
    except ValueError: out["referable_auroc"] = np.nan
    return out

def bootstrap_grading_metrics(pred_df, n=CFG.BOOTSTRAP_N):
    d = pred_df[pred_df.y_true >= 0].reset_index(drop=True)
    if len(d) == 0: return {}
    rng = np.random.default_rng(CFG.SEED)
    vals = defaultdict(list)
    for _ in range(n):
        b = d.iloc[rng.integers(0, len(d), len(d))]
        m = grading_metrics(b)
        for k, v in m.items():
            if k != "n" and np.isfinite(v): vals[k].append(v)
    return {k: {
        "mean": float(np.mean(v)),
        "ci_low": float(np.quantile(v, .025)),
        "ci_high": float(np.quantile(v, .975))
    } for k, v in vals.items()}

grading_results = {}
if idrid_df.label.notna().any():
    grade_df = idrid_df.copy()
    if "task" in grade_df.columns:
        grade_df = grade_df[grade_df.task == "GRADE"]
    grade_df = grade_df[grade_df.label.notna()].copy()
    print(f"Grading evaluation on {len(grade_df)} disease-grading images "
          f"(train={int((grade_df.split=='TRAIN').sum()) if 'split' in grade_df else '?'}, "
          f"test={int((grade_df.split=='TEST').sum()) if 'split' in grade_df else '?'})")
    for name, model in MODELS.items():
        pred = predict_frame(model, grade_df)
        pred.to_csv(CFG.TABLE_DIR / f"idrid_grading_predictions_{name.lower()}.csv", index=False)
        grading_results[name] = grading_metrics(pred)
        grading_results[name]["bootstrap_95ci"] = bootstrap_grading_metrics(pred)
        print(name, grading_results[name])

        cm = confusion_matrix(pred.y_true, pred.y_pred, labels=range(5))
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set(title=f"{name} — IDRiD", xlabel="Predicted grade", ylabel="True grade")
        fig.tight_layout()
        fig.savefig(CFG.FIG_DIR / f"idrid_cm_{name.lower()}.png", dpi=300)
        fig.savefig(CFG.FIG_DIR / f"idrid_cm_{name.lower()}.pdf")
        plt.show()
    pd.DataFrame(grading_results).T.to_csv(CFG.TABLE_DIR / "idrid_grading_metrics.csv")

    # Paired comparison on the same IDRiD eyes: exact McNemar + bootstrap QWK delta.
    b = pd.read_csv(CFG.TABLE_DIR/"idrid_grading_predictions_baseline.csv")
    e = pd.read_csv(CFG.TABLE_DIR/"idrid_grading_predictions_enhanced.csv")
    paired = b[["image_id","y_true","y_pred"]].merge(
        e[["image_id","y_pred"]], on="image_id", suffixes=("_baseline","_enhanced")
    )
    bc = (paired.y_pred_baseline == paired.y_true)
    ec = (paired.y_pred_enhanced == paired.y_true)
    n01 = int((~bc & ec).sum()); n10 = int((bc & ~ec).sum())
    mcnemar_p = float(binomtest(min(n01,n10), n01+n10, .5).pvalue) if n01+n10 else 1.0
    rng = np.random.default_rng(CFG.SEED); deltas=[]
    for _ in range(CFG.BOOTSTRAP_N):
        s = paired.iloc[rng.integers(0,len(paired),len(paired))]
        try:
            qb = cohen_kappa_score(s.y_true,s.y_pred_baseline,weights="quadratic")
            qe = cohen_kappa_score(s.y_true,s.y_pred_enhanced,weights="quadratic")
            if np.isfinite(qb) and np.isfinite(qe): deltas.append(qe-qb)
        except Exception: pass
    paired_stats = {
        "n":len(paired),"mcnemar_n01_baseline_wrong_enhanced_correct":n01,
        "mcnemar_n10_baseline_correct_enhanced_wrong":n10,"mcnemar_exact_p":mcnemar_p,
        "qwk_delta_enhanced_minus_baseline":float(np.mean(deltas)) if deltas else np.nan,
        "qwk_delta_ci_low":float(np.quantile(deltas,.025)) if deltas else np.nan,
        "qwk_delta_ci_high":float(np.quantile(deltas,.975)) if deltas else np.nan,
    }
    pd.DataFrame([paired_stats]).to_csv(CFG.TABLE_DIR/"idrid_paired_model_statistics.csv",index=False)
    print("Paired model statistics:", paired_stats)
else:
    print("No grade labels matched. Lesion-grounded XAI will still run.")
"""),

code(r"""# ========================= TRAINING CURVES + ROC (FROM SAVED HISTORIES) =========================
def plot_history(path, name):
    if not Path(path).exists():
        print("History missing:", path); return
    h = pd.read_csv(path)
    if "epoch" not in h: h["epoch"] = np.arange(1, len(h)+1)
    rename = {}
    if "qwk" in h and "val_qwk" not in h: rename["qwk"] = "val_qwk"
    if "loss" in h and "train_loss" not in h: rename["loss"] = "train_loss"
    h = h.rename(columns=rename)
    h.to_csv(CFG.TABLE_DIR / f"{name.lower()}_history.csv", index=False)

    fig, axes = plt.subplots(1,2,figsize=(9,3.5))
    for col, label in [("train_loss","Train"),("val_loss","Validation")]:
        if col in h: axes[0].plot(h.epoch,h[col],marker="o",label=label)
    for col, label in [("train_qwk","Train"),("val_qwk","Validation")]:
        if col in h: axes[1].plot(h.epoch,h[col],marker="o",label=label)
    axes[0].set(title=f"{name}: loss",xlabel="Epoch",ylabel="Loss")
    axes[1].set(title=f"{name}: QWK",xlabel="Epoch",ylabel="QWK")
    for ax in axes: ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    fig.savefig(CFG.FIG_DIR / f"training_curves_{name.lower()}.png", dpi=300)
    fig.savefig(CFG.FIG_DIR / f"training_curves_{name.lower()}.pdf")
    plt.show()

plot_history(CFG.OUT_DIR/"M0_history.csv","Baseline")
plot_history(CFG.OUT_DIR/"M2_history.csv","Enhanced")

if idrid_df.label.notna().any():
    fig, ax = plt.subplots(figsize=(5,4))
    for name in MODELS:
        p = pd.read_csv(CFG.TABLE_DIR/f"idrid_grading_predictions_{name.lower()}.csv")
        y = (p.y_true>=2).astype(int)
        if y.nunique()<2: continue
        fpr,tpr,_=roc_curve(y,p.referable_score)
        ax.plot(fpr,tpr,lw=2,label=f"{name} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0,1],[0,1],"k--",alpha=.5)
    ax.set(xlabel="False-positive rate",ylabel="True-positive rate",title="IDRiD referable DR ROC")
    ax.legend(); fig.tight_layout()
    fig.savefig(CFG.FIG_DIR/"idrid_referable_roc.png",dpi=300)
    fig.savefig(CFG.FIG_DIR/"idrid_referable_roc.pdf")
    plt.show()
"""),

code(r"""# ========================= XAI METHODS: ROLLOUT, ViT GRAD-CAM, IG =========================
def normalize_map(m):
    m = np.asarray(m, dtype=np.float32)
    m = np.nan_to_num(m)
    m -= m.min()
    return m / (m.max() + 1e-8)

def resize_map(m, size=CFG.IMG_SIZE):
    im = Image.fromarray(np.uint8(normalize_map(m)*255))
    return np.asarray(im.resize((size,size), Image.BILINEAR)).astype(np.float32)/255

def attention_rollout(model, x):
    # Compute rollout from qkv inputs; no model modification required.
    backbone = model.backbone
    saved = {}
    handles = []
    for i, block in enumerate(backbone.blocks):
        qkv = block.attn.qkv
        handles.append(qkv.register_forward_pre_hook(
            lambda module, inputs, idx=i: saved.__setitem__(idx, inputs[0].detach())
        ))
    with torch.no_grad():
        _ = logits_of(model, x)
        joint = None
        for i, block in enumerate(backbone.blocks):
            tokens = saved[i]
            qkv_layer = block.attn.qkv
            qkv = qkv_layer(tokens)
            B,N,C3 = qkv.shape
            heads = block.attn.num_heads
            qkv = qkv.reshape(B,N,3,heads,C3//(3*heads)).permute(2,0,3,1,4)
            q,k = qkv[0],qkv[1]
            attn = (q @ k.transpose(-2,-1)) * block.attn.scale
            attn = attn.softmax(dim=-1).mean(dim=1)[0]
            attn = attn + torch.eye(N,device=attn.device)
            attn = attn / attn.sum(dim=-1,keepdim=True)
            joint = attn if joint is None else attn @ joint
        cam = joint[0,1:].reshape(14,14).detach().cpu().numpy()
    for h in handles: h.remove()
    return resize_map(cam)

def vit_gradcam(model, x, target_class=None, block_idx=23):
    captured = {}
    def hook(_m,_i,out):
        captured["act"] = out
        out.retain_grad()
    h = model.backbone.blocks[block_idx].register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    xg = x.detach().requires_grad_(True)
    logits = logits_of(model,xg)
    target_class = int(logits.argmax(1).item()) if target_class is None else int(target_class)
    logits[0,target_class].backward()
    act = captured["act"][0,1:]
    grad = captured["act"].grad[0,1:]
    weights = grad.mean(dim=0,keepdim=True)
    cam = F.relu((act*weights).sum(dim=1)).reshape(14,14)
    h.remove()
    return resize_map(cam.detach().cpu().numpy()), target_class

def integrated_gradients(model, x, target_class=None, steps=CFG.IG_STEPS):
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        target_class = int(logits_of(model,x).argmax(1).item()) if target_class is None else int(target_class)
    baseline = torch.zeros_like(x)
    total_grad = torch.zeros_like(x)
    for alpha in torch.linspace(0,1,steps,device=x.device):
        xi = (baseline + alpha*(x-baseline)).detach().requires_grad_(True)
        score = logits_of(model,xi)[0,target_class]
        grad = torch.autograd.grad(score,xi,retain_graph=False)[0]
        total_grad += grad.detach()
    attr = (x-baseline)*total_grad/steps
    sal = attr.abs().sum(dim=1)[0].detach().cpu().numpy()
    return resize_map(sal), target_class

def overlay(rgb, heat, alpha=.45):
    cmap = plt.get_cmap("jet")(normalize_map(heat))[...,:3]
    return np.clip((1-alpha)*rgb + alpha*cmap,0,1)

# One-image smoke test (fails early if hooks/checkpoints are incompatible).
row = xai_df.iloc[0]
x,_,rgb = preprocess_pil(Image.open(row.image_path))
x = x.unsqueeze(0).to(DEVICE)
for name,model in MODELS.items():
    roll = attention_rollout(model,x)
    gc, pred = vit_gradcam(model,x)
    print(name, "predicted grade", pred, "rollout", roll.shape, "gradcam", gc.shape)
print("XAI smoke test passed.")
"""),

code(r"""# ========================= LESION ALIGNMENT EXPERIMENT =========================
def overlap_metrics(heat, mask, top_frac):
    heat = normalize_map(heat)
    mask = np.asarray(mask).astype(bool)
    threshold = np.quantile(heat, 1-top_frac)
    pred = heat >= threshold
    inter = np.logical_and(pred,mask).sum()
    union = np.logical_or(pred,mask).sum()
    return {
        "pointing": float(mask[np.unravel_index(np.argmax(heat),heat.shape)]) if mask.any() else np.nan,
        "energy": float(heat[mask].sum()/(heat.sum()+1e-8)) if mask.any() else np.nan,
        "iou": float(inter/(union+1e-8)) if mask.any() else np.nan,
        "dice": float(2*inter/(pred.sum()+mask.sum()+1e-8)) if mask.any() else np.nan,
    }

records = []
qual_examples = []
for idx,row in xai_df.iterrows():
    x,y,image_id,rgb,masks,lesion = IDRiDDataset(xai_df,True)[idx]
    xb = x.unsqueeze(0).to(DEVICE)
    for model_name, model in MODELS.items():
        with torch.no_grad():
            prob = torch.softmax(logits_of(model,xb),1)[0].cpu().numpy()
        pred = int(prob.argmax())
        method_maps = {
            "AttentionRollout": attention_rollout(model,xb),
            "GradCAM": vit_gradcam(model,xb,pred)[0],
            "RandomControl": np.random.default_rng(CFG.SEED + idx).random(
                (CFG.IMG_SIZE, CFG.IMG_SIZE), dtype=np.float32
            ),
        }
        if masks["OD"].any():
            method_maps["OpticDiscControl"] = normalize_map(gaussian_filter(masks["OD"].astype(float), sigma=8))
        if CFG.RUN_INTEGRATED_GRADIENTS and idx < CFG.IG_MAX_IMAGES:
            method_maps["IntegratedGradients"] = integrated_gradients(model,xb,pred)[0]

        for method,heat in method_maps.items():
            np.save(CFG.HEATMAP_DIR/f"{image_id}_{model_name}_{method}.npy",heat.astype(np.float16))
            for mask_type,mask in {**masks,"LESION_UNION":lesion}.items():
                if not mask.any(): continue
                for top_frac in CFG.TOP_K:
                    met = overlap_metrics(heat,mask,top_frac)
                    records.append({
                        "image_id":image_id,"model":model_name,"method":method,
                        "mask_type":mask_type,"top_frac":top_frac,
                        "y_true":y,"y_pred":pred,"correct":int(y==pred) if y>=0 else np.nan,
                        "confidence":float(prob[pred]),**met
                    })
        if idx < 6:
            qual_examples.append((image_id,model_name,rgb,lesion,method_maps,pred,y))
    print(f"[{idx+1:03d}/{len(xai_df)}] {image_id}")

alignment_df = pd.DataFrame(records)
alignment_df.to_csv(CFG.TABLE_DIR/"lesion_alignment_per_image.csv",index=False)
print("Alignment rows:",len(alignment_df))
display(alignment_df.head())

# Qualitative paper panels.
for image_id,model_name,rgb,lesion,maps,pred,y in qual_examples:
    n = 2+len(maps)
    fig,axes=plt.subplots(1,n,figsize=(3*n,3))
    axes[0].imshow(rgb); axes[0].set_title(f"{image_id}\nTrue={y}, Pred={pred}")
    axes[1].imshow(lesion,cmap="gray"); axes[1].set_title("Lesion union")
    for ax,(method,heat) in zip(axes[2:],maps.items()):
        ax.imshow(overlay(rgb,heat)); ax.set_title(method)
    for ax in axes: ax.axis("off")
    fig.suptitle(model_name); fig.tight_layout()
    fig.savefig(CFG.FIG_DIR/f"qual_{image_id}_{model_name.lower()}.png",dpi=300,bbox_inches="tight")
    plt.show()
"""),

code(r"""# ========================= BOOTSTRAP CIs + PAIRED TESTS + PAPER PLOTS =========================
def bootstrap_ci(values,n=CFG.BOOTSTRAP_N,seed=CFG.SEED):
    v=np.asarray(pd.Series(values).dropna(),dtype=float)
    if len(v)==0:return (np.nan,np.nan,np.nan)
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(n)])
    return float(v.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))

union = alignment_df[alignment_df.mask_type=="LESION_UNION"].copy()
summary=[]
for keys,g in union.groupby(["model","method","top_frac"]):
    for metric in ["pointing","energy","iou","dice"]:
        mean,lo,hi=bootstrap_ci(g[metric])
        summary.append(dict(zip(["model","method","top_frac"],keys),
                            metric=metric,n=g[metric].notna().sum(),mean=mean,ci_low=lo,ci_high=hi))
summary_df=pd.DataFrame(summary)
summary_df.to_csv(CFG.TABLE_DIR/"lesion_alignment_summary_ci.csv",index=False)
display(summary_df)

# Per-lesion summary.
per_lesion=(alignment_df.groupby(["model","method","mask_type","top_frac"])
            [["pointing","energy","iou","dice"]].agg(["count","mean","std"]).reset_index())
per_lesion.to_csv(CFG.TABLE_DIR/"per_lesion_alignment_summary.csv",index=False)

# Paired Baseline vs Enhanced, same image/method/threshold.
tests=[]
for (method,top_frac),g in union.groupby(["method","top_frac"]):
    for metric in ["pointing","energy","iou","dice"]:
        w=g.pivot_table(index="image_id",columns="model",values=metric,aggfunc="mean").dropna()
        if {"Baseline","Enhanced"}.issubset(w.columns) and len(w)>=5:
            try: stat,p=wilcoxon(w["Enhanced"],w["Baseline"],zero_method="zsplit")
            except ValueError: stat,p=np.nan,np.nan
            tests.append({"method":method,"top_frac":top_frac,"metric":metric,"n":len(w),
                          "baseline_mean":w.Baseline.mean(),"enhanced_mean":w.Enhanced.mean(),
                          "mean_delta":(w.Enhanced-w.Baseline).mean(),
                          "wilcoxon_stat":stat,"p_value":p})
tests_df=pd.DataFrame(tests)
tests_df.to_csv(CFG.TABLE_DIR/"paired_baseline_vs_enhanced_wilcoxon.csv",index=False)
display(tests_df)

# Correct vs incorrect (where grade labels available).
if union.correct.notna().any():
    strata=(union.groupby(["model","method","top_frac","correct"])
            [["pointing","energy","iou","dice"]].agg(["count","mean","std"]).reset_index())
    strata.to_csv(CFG.TABLE_DIR/"alignment_correct_vs_incorrect.csv",index=False)

# Bar plot with 95% CI.
plot_df=summary_df[(summary_df.top_frac==0.20)&summary_df.metric.isin(["energy","iou","dice"])]
for metric in ["energy","iou","dice"]:
    d=plot_df[plot_df.metric==metric].copy()
    if len(d)==0:continue
    fig,ax=plt.subplots(figsize=(8,4))
    d["label"]=d["model"]+" | "+d["method"]
    yerr=np.vstack([d["mean"]-d["ci_low"],d["ci_high"]-d["mean"]])
    ax.bar(np.arange(len(d)),d["mean"],yerr=yerr,capsize=3)
    ax.set_xticks(np.arange(len(d)));ax.set_xticklabels(d.label,rotation=35,ha="right")
    ax.set_ylabel(metric.title());ax.set_title(f"Lesion-union {metric} (top 20%; bootstrap 95% CI)")
    fig.tight_layout()
    fig.savefig(CFG.FIG_DIR/f"alignment_{metric}_ci.png",dpi=300,bbox_inches="tight")
    fig.savefig(CFG.FIG_DIR/f"alignment_{metric}_ci.pdf",bbox_inches="tight")
    plt.show()
"""),

code(r"""# ========================= DELETION / INSERTION FAITHFULNESS =========================
def score_target(model,x,target):
    with torch.no_grad():
        return float(torch.softmax(logits_of(model,x),1)[0,target].item())

def perturbation_curve(model,x,heat,target,mode="deletion",steps=CFG.PERTURB_STEPS):
    # Perturb 14x14 patch grid in ranked order; baseline is zero (ImageNet mean after normalization).
    order=np.argsort(resize_map(heat,14).ravel())[::-1]
    patch=CFG.IMG_SIZE//14
    base=torch.zeros_like(x)
    current=x.clone() if mode=="deletion" else base.clone()
    scores=[score_target(model,current,target)]
    chunk=int(math.ceil(len(order)/steps))
    for s in range(steps):
        for idx in order[s*chunk:(s+1)*chunk]:
            r,c=divmod(int(idx),14)
            sl=(slice(None),slice(None),slice(r*patch,(r+1)*patch),slice(c*patch,(c+1)*patch))
            current[sl]=base[sl] if mode=="deletion" else x[sl]
        scores.append(score_target(model,current,target))
    fractions=np.linspace(0,1,len(scores))
    return fractions,np.array(scores)

faith_records=[]
if CFG.RUN_FAITHFULNESS:
    faith_df=xai_df.head(CFG.FAITHFULNESS_MAX_IMAGES)
    ds=IDRiDDataset(faith_df,True)
    for idx,row in faith_df.iterrows():
        x,y,image_id,rgb,masks,lesion=ds[idx]
        xb=x.unsqueeze(0).to(DEVICE)
        for model_name,model in MODELS.items():
            with torch.no_grad(): pred=int(logits_of(model,xb).argmax(1).item())
            maps={
                "AttentionRollout":attention_rollout(model,xb),
                "GradCAM":vit_gradcam(model,xb,pred)[0],
                "RandomControl":np.random.default_rng(CFG.SEED+idx).random(
                    (CFG.IMG_SIZE,CFG.IMG_SIZE),dtype=np.float32
                ),
            }
            if masks["OD"].any():
                maps["OpticDiscControl"]=normalize_map(gaussian_filter(masks["OD"].astype(float),sigma=8))
            if CFG.RUN_INTEGRATED_GRADIENTS and idx<CFG.IG_MAX_IMAGES:
                maps["IntegratedGradients"]=integrated_gradients(model,xb,pred)[0]
            for method,heat in maps.items():
                for mode in ["deletion","insertion"]:
                    frac,scores=perturbation_curve(model,xb,heat,pred,mode)
                    auc_score=float(np.trapz(scores,frac))
                    faith_records.append({"image_id":image_id,"model":model_name,"method":method,
                                          "mode":mode,"auc":auc_score,"fractions":json.dumps(frac.tolist()),
                                          "scores":json.dumps(scores.tolist())})
        print("Faithfulness",idx+1,"/",len(faith_df),image_id)

faith_df=pd.DataFrame(faith_records)
if len(faith_df):
    faith_df.to_csv(CFG.TABLE_DIR/"faithfulness_per_image.csv",index=False)
    faith_summary=(faith_df.groupby(["model","method","mode"]).auc.agg(["count","mean","std"]).reset_index())
    faith_summary.to_csv(CFG.TABLE_DIR/"faithfulness_summary.csv",index=False)
    display(faith_summary)

    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,mode in zip(axes,["deletion","insertion"]):
        for (model,method),g in faith_df[faith_df["mode"]==mode].groupby(["model","method"]):
            curves=np.vstack([json.loads(s) for s in g.scores])
            frac=np.array(json.loads(g.fractions.iloc[0]))
            ax.plot(frac,curves.mean(0),label=f"{model}-{method}")
        ax.set(title=mode.title(),xlabel="Fraction of patches perturbed",ylabel="Target probability")
        ax.legend(fontsize=7);ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(CFG.FIG_DIR/"faithfulness_curves.png",dpi=300)
    fig.savefig(CFG.FIG_DIR/"faithfulness_curves.pdf")
    plt.show()
"""),

code(r"""# ========================= FINAL MANIFEST + STATISTICAL SUMMARY + ZIP =========================
summary = {
    "device": str(DEVICE),
    "seed": CFG.SEED,
    "idrid_root": str(IDRID_ROOT),
    "weights_path": WEIGHTS_PATH,
    "saved_outputs": str(SAVED_OUT),
    "idrid_images": int(len(idrid_df)),
    "lesion_positive_images": int((idrid_df.n_lesion_masks>0).sum()),
    "xai_images": int(len(xai_df)),
    "grading_metrics": grading_results,
    "xai_methods": sorted(alignment_df.method.unique().tolist()),
    "top_k": list(CFG.TOP_K),
    "bootstrap_n": CFG.BOOTSTRAP_N,
    "notes": [
        "Frozen checkpoint evaluation; no retraining.",
        "Optic disc excluded from lesion union and retained as a control mask.",
        "Report both clinical alignment and model faithfulness; neither alone proves causality."
    ]
}
with open(CFG.OUT_DIR/"experiment_summary.json","w") as f:
    json.dump(summary,f,indent=2)

manifest=[]
for p in CFG.OUT_DIR.rglob("*"):
    if p.is_file():
        manifest.append({"file":str(p.relative_to(CFG.OUT_DIR)),"bytes":p.stat().st_size})
pd.DataFrame(manifest).to_csv(CFG.OUT_DIR/"manifest.csv",index=False)

zip_path=Path("/kaggle/working/RETFound_IDRiD_XAI_Journal_Outputs.zip") if CFG.IS_KAGGLE else Path("RETFound_IDRiD_XAI_Journal_Outputs.zip")
with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
    for p in CFG.OUT_DIR.rglob("*"):
        if p.is_file(): z.write(p,p.relative_to(CFG.OUT_DIR))

print("\n=== PAPER-READY STATISTICAL SUMMARY ===")
print(json.dumps(summary,indent=2))
print("\nKey tables:")
for p in sorted(CFG.TABLE_DIR.glob("*.csv")): print(" -",p.name)
print("\nFigures:",len(list(CFG.FIG_DIR.glob("*.png"))),"PNG +",len(list(CFG.FIG_DIR.glob("*.pdf"))),"PDF")
print("Download:",zip_path)
"""),

md(r"""## Reporting checklist after the run

Before drafting the IEEE journal paper, verify:

- IDRiD image/mask counts match the source documentation.
- Grade labels are correctly matched by image ID (if the mirror includes grading CSVs).
- Every method beats random/optic-disc controls before claiming lesion alignment.
- Report paired Baseline-vs-Enhanced tests on the same images.
- Report both clinical overlap and deletion/insertion faithfulness.
- Avoid causal language: heatmaps are post-hoc evidence, not proof of reasoning.

For the full journal package, run a second lesion dataset (DDR/FGADR) and add a clinician review if feasible.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kaggle": {"accelerator": "gpu", "isGpuEnabled": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells)")
