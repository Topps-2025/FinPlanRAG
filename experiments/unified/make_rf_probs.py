"""Train the non-LLM RF belief model on dev and write validation/test probabilities."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_real_benchmark import build, company_key, load_records  # noqa: E402

CLASSES = ["below", "within", "above"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO_ROOT / "paper/data/dataset_v4.json"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--train-splits", default="dev")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = load_records(Path(args.data))
    by_company: dict[str, list] = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    train_splits = {x.strip() for x in args.train_splits.split(",") if x.strip()}
    train = [row for row in rows if row[2].get("split") in train_splits]
    target = [row for row in rows if row[2].get("split") == args.split]
    Xd, Cd, yd, _ = build(train, by_company)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    Xc = encoder.fit_transform(Cd)
    X = np.hstack([Xd, Xc])
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                   class_weight="balanced_subsample", random_state=1, n_jobs=-1)
    model.fit(X, list(map(str, yd)))
    Xt, Ct, yt, meta = build(target, by_company)
    Xtest = np.hstack([Xt, encoder.transform(Ct)])
    probs = model.predict_proba(Xtest)
    order = [list(model.classes_).index(c) for c in CLASSES]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for rec, truth, p in zip(meta, yt, probs):
            handle.write(json.dumps({
                "claim_id": rec["claim_id"], "truth": str(truth),
                "probabilities": [float(p[i]) for i in order],
            }, ensure_ascii=False) + "\n")
    print("wrote", len(meta), "to", out)


if __name__ == "__main__":
    main()
