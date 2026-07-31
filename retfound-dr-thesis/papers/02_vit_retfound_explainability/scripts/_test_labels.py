"""Offline check that grading labels come from the official IDRiD CSVs, not community mirrors."""
import csv as csvmod
import json, re, shutil, tempfile, types
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "RETFound_IDRiD_Explainability_Journal_Kaggle.ipynb"
CELLS = [
    "".join(c["source"])
    for c in json.load(open(NB, encoding="utf-8"))["cells"]
    if c["cell_type"] == "code"
]
CELL = next(c for c in CELLS if "OFFICIAL_GRADE_COLS" in c)
BLOCK = CELL[CELL.index("# Grading labels: require the OFFICIAL"):]
BLOCK = BLOCK[:BLOCK.index("# Persist provenance")]


class FakeDF:
    def __init__(self, rows, cols):
        self.rows = [{c: r.get(c, "") for c in cols} for r in rows]
        self._cols = list(cols)

    @property
    def columns(self):
        return self._cols

    @columns.setter
    def columns(self, value):
        old, new = list(self._cols), list(value)
        remapped = []
        for r in self.rows:
            remapped.append({new[i]: r.get(old[i], "") for i in range(len(new))})
        self.rows, self._cols = remapped, new

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, list):
            return FakeDF(self.rows, key)
        raise TypeError(key)

    def iterrows(self):
        return enumerate(self.rows)


def read_csv(path, nrows=None):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csvmod.DictReader(fh))
    cols = list(rows[0].keys()) if rows else []
    return FakeDF(rows if nrows is None else rows[:nrows], cols)


def write_csv(path, cols, n, start=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csvmod.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i in range(start, start + n):
            row = {c: "" for c in cols}
            row[cols[0]] = f"IDRiD_{i:03d}"
            for c in cols:
                if "grade" in c.lower():
                    row[c] = str(i % 5)
            w.writerow(row)


OFFICIAL = ["Image name", "Retinopathy grade", "Risk of macular edema"]
MIRROR = OFFICIAL + ["class", "caption"]


def run_case(name, files, expect_source, expect_n):
    tmp = Path(tempfile.mkdtemp())
    inp = tmp / "input"
    for rel, cols, n, start in files:
        write_csv(inp / rel, cols, n, start=start)
    out = tmp / "out"
    out.mkdir()

    ns = {
        "Path": Path, "re": re, "json": json,
        "pd": types.SimpleNamespace(read_csv=read_csv),
        "CFG": types.SimpleNamespace(INPUT_ROOT=inp, OUT_DIR=out),
        "all_files": lambda root: [p for p in Path(root).rglob("*") if p.is_file()],
        "OTHER_DATASET_KEYS": ("messidor", "aptos", "eyepacs"),
        "canonical_id": lambda p, split=None, task=None: (
            f"{task or 'GRADE'}_{split or 'UNK'}_{Path(p).stem.upper()}"
        ),
        "infer_split": lambda p: "TRAIN" if "train" in str(p).lower() else (
            "TEST" if "test" in str(p).lower() else "UNK"
        ),
        "print": lambda *a, **k: None,
    }
    exec(BLOCK, ns)
    src, n = ns["GRADE_LABEL_SOURCE"], len(ns["labels"])
    ok = src == expect_source and n == expect_n
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       -> source={src}, labels={n} "
          f"(expected {expect_source}, {expect_n})")
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


MIRROR_CSV = ("datasets/mohamedabdalkader/idrid/IDRiD/Train/annotations.csv", MIRROR, 1239, 1)
TRAIN_CSV = ("datasets/official/B. Disease Grading/IDRiD_Disease_Grading_Training_Labels.csv",
             OFFICIAL, 413, 1)
TEST_CSV = ("datasets/official/B. Disease Grading/IDRiD_Disease_Grading_Testing_Labels.csv",
            OFFICIAL, 103, 1)  # same numeric stems as train; split must keep them distinct

results = [
    run_case("caption mirror only -> rejected, flagged as fallback",
             [MIRROR_CSV], "non-official-fallback", 1239),
    run_case("official train+test only -> accepted, 516 labels",
             [TRAIN_CSV, TEST_CSV], "official", 516),
    run_case("both attached -> official wins, mirror ignored",
             [MIRROR_CSV, TRAIN_CSV, TEST_CSV], "official", 516),
]
print("\nALL PASS" if all(results) else "\nFAILURES PRESENT")
