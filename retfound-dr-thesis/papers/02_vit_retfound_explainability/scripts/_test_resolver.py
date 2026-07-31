"""Offline check that input resolution picks the right IDRiD root under nested mounts."""
import json, re, shutil, tempfile, types
from pathlib import Path
from dataclasses import dataclass

# pandas cannot load in this local environment; only DataFrame(...) for display is needed.
pd = types.SimpleNamespace(DataFrame=lambda rows: types.SimpleNamespace(
    sort_values=lambda *a, **k: types.SimpleNamespace(head=lambda n: rows)))

NB = Path(__file__).resolve().parents[1] / "notebooks" / "RETFound_IDRiD_Explainability_Journal_Kaggle.ipynb"
CELL = next(
    "".join(c["source"])
    for c in json.load(open(NB, encoding="utf-8"))["cells"]
    if c["cell_type"] == "code" and "resolve_idrid_root" in "".join(c["source"])
)
PREP = CELL.split("IDRID_ROOT = resolve_idrid_root()")[0]


def make_tree(root, files):
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def run_case(name, files, expect_substr, expect_masks_gt0):
    tmp = Path(tempfile.mkdtemp())
    inp = tmp / "input"
    make_tree(inp, files)

    @dataclass
    class CFG:
        INPUT_ROOT = inp

    ns = {"Path": Path, "re": re, "pd": pd, "shutil": shutil,
          "CFG": CFG, "display": lambda *a, **k: None, "print": lambda *a, **k: None}
    exec(PREP, ns)
    root = ns["resolve_idrid_root"]()
    masks = ns["mask_like_count"](root)
    ok = expect_substr in str(root).replace("\\", "/") and (masks > 0) == expect_masks_gt0
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       -> {Path(root).name}  mask_like={masks}")
    shutil.rmtree(tmp, ignore_errors=True)
    return ok

# Reproduces the reported layout: IDRiD grading mirror beside Messidor-2 under /input/datasets/<owner>/<slug>
grading_only = (
    [f"datasets/mariaherrerot/messidor2preprocess/images/{i}_PP.png" for i in range(30)]
    + ["datasets/mohamedabdalkader/indian-diabetic-retinopathy-image-dataset-idrid/IDRiD/Train/annotations.csv"]
    + [f"datasets/mohamedabdalkader/indian-diabetic-retinopathy-image-dataset-idrid/IDRiD/Train/IDRiD_{i:03d}.jpg"
       for i in range(20)]
)

# Audited Kaggle mirror: aaryapatel98 with official A. Segmentation layout.
aarya = grading_only + [
    f"datasets/aaryapatel98/indian-diabetic-retinopathy-image-dataset/A. Segmentation/"
    f"1. Original Images/a. Training Set/IDRiD_{i:02d}.jpg"
    for i in range(10)
] + [
    f"datasets/aaryapatel98/indian-diabetic-retinopathy-image-dataset/A. Segmentation/"
    f"2. All Segmentation Groundtruths/a. Training Set/{d}/IDRiD_{i:02d}_{c}.tif"
    for i in range(10)
    for d, c in [("1. Microaneurysms", "MA"), ("2. Haemorrhages", "HE"),
                 ("3. Hard Exudates", "EX"), ("4. Soft Exudates", "SE"),
                 ("5. Optic Disc", "OD")]
]

results = [
    run_case("grading-only mirror next to Messidor-2", grading_only,
             "indian-diabetic-retinopathy-image-dataset-idrid", False),
    run_case("aaryapatel98 + grading mirror both attached", aarya,
             "aaryapatel98", True),
]
print("\nALL PASS" if all(results) else "\nFAILURES PRESENT")
