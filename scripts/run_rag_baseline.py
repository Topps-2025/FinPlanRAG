"""Retrieval-augmented reader baseline over point-in-time historical claims.

For each test fold, retrieve the most similar earlier claims from other
companies (TF-IDF over claim text plus guidance metadata), put their public
evidence and realized outcomes in the prompt, and ask the same base model for a
label, confidence, and credible set. No test outcome is retrieved or shown.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from deepseek_client import DeepSeekClient, DeepSeekError
from run_company_disjoint_benchmark import FOLDS, TEST_YEARS, fold_of
from run_deepseek_baselines import SYSTEM, history_for
from run_real_benchmark import LABELS, company_key, load_records, parse_date

REPO_ROOT = Path(__file__).resolve().parents[1]
# Released benchmark copy. Override with --data if needed.
DEFAULT_DATA = REPO_ROOT / "paper" / "data" / "dataset_v4.json"


def serialize(record: dict[str, Any], by_company: dict[str, list]) -> str:
    low = float(record["guidance_low_musd"])
    high = float(record["guidance_high_musd"])
    mid = float(record["guidance_mid_musd"])
    current = float(record["current_q_musd"])
    claim = str(record.get("claim_text") or "").replace("\n", " ")[:420]
    return (
        f"Cutoff: {record.get('cutoff')}; sector: {record.get('sector') or 'other'}; "
        f"claim: {claim}; guided ${low:.1f}M-${high:.1f}M "
        f"(mid ${mid:.1f}M); current quarter ${current:.1f}M; "
        f"{history_for(record, by_company)}"
    )


def retrieve(train_rows, test_row, by_company, k: int, mode: str = "text"):
    train_text = [str(record.get("claim_text") or "") for _, _, record in train_rows]
    test_text = [str(test_row.get("claim_text") or "")]
    vectorizer = TfidfVectorizer(
        max_features=4000, ngram_range=(1, 2), min_df=2, stop_words="english"
    )
    matrix = vectorizer.fit_transform(train_text)
    query = vectorizer.transform(test_text)
    scores = cosine_similarity(query, matrix).ravel()
    if mode == "multi":
        test_width = float(test_row.get("guidance_width_pct") or 0.0)
        test_mid = float(test_row.get("guidance_mid_musd") or 0.0)
        test_current = float(test_row.get("current_q_musd") or 0.0)
        test_ratio = test_current / test_mid if test_mid else 0.0
        test_sector = str(test_row.get("sector") or "other")
        numeric = []
        for _, _, record in train_rows:
            width = float(record.get("guidance_width_pct") or 0.0)
            mid = float(record.get("guidance_mid_musd") or 0.0)
            current = float(record.get("current_q_musd") or 0.0)
            ratio = current / mid if mid else 0.0
            sector_bonus = 1.0 if str(record.get("sector") or "other") == test_sector else 0.0
            numeric.append(
                np.exp(-abs(width - test_width) * 10.0)
                + np.exp(-abs(ratio - test_ratio) * 4.0)
                + 0.5 * sector_bonus
            )
        scores = scores + 0.15 * np.asarray(numeric, dtype=float)
    order = np.argsort(-scores)[:k]
    return [(train_rows[i], float(scores[i])) for i in order]


def prompt_for(record, retrieved, by_company) -> str:
    lines = [
        "Decide whether the realized revenue for the target period will be "
        "below, within, or above the guided range. Use only the cutoff evidence "
        "and the retrieved earlier claims. Return only JSON.",
        "",
        "Current claim:",
        serialize(record, by_company),
        "",
        "Retrieved earlier claims from other companies (their realized outcomes are public):",
    ]
    for index, ((_, _, item), score) in enumerate(retrieved, start=1):
        actual = str(item["outcome"]["bucket"])
        lines.append(
            f"{index}. similarity={score:.3f}; actual outcome={actual}; "
            f"{serialize(item, by_company)}"
        )
    lines += [
        "",
        "Return only: {\"label\":\"below|within|above\",\"confidence\":0.0,"
        "\"credible_set\":[\"below|within|above\"],\"reason\":\"at most 25 words\"}",
    ]
    return "\n".join(lines)


def parse_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def run_one(client, record, train_rows, by_company, k, mode, max_tokens):
    retrieved = retrieve(train_rows, record, by_company, k, mode)
    prompt = prompt_for(record, retrieved, by_company)
    started = time.time()
    try:
        response = client.complete(
            prompt,
            system=SYSTEM,
            max_tokens=max_tokens,
            temperature=0.0,
            thinking=False,
            use_cache=True,
        )
        text = client.text_of(response)
        parsed = parse_json(text)
        return {
            "claim_id": record["claim_id"],
            "variant": "rag_knn",
            "truth": str(record["outcome"]["bucket"]),
            "parsed": parsed,
            "error": None if parsed else "parse_failed",
            "raw_text": text[:1200],
            "usage": client.usage_of(response),
            "latency_seconds": time.time() - started,
            "retrieved_ids": [str(item["claim_id"]) for (_, _, item), _ in retrieved],
        }
    except DeepSeekError as error:
        return {
            "claim_id": record["claim_id"],
            "variant": "rag_knn",
            "truth": str(record["outcome"]["bucket"]),
            "parsed": None,
            "error": f"api_error:{error}",
            "raw_text": "",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "latency_seconds": time.time() - started,
            "retrieved_ids": [],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", choices=("test", "dev_validation"), default="test")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--mode", choices=("text", "multi"), default="text")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = load_records(args.data)
    by_company = defaultdict(list)
    for row in rows:
        by_company[company_key(row[2])].append(row)
    if args.split == "test":
        selected = [
            row
            for row in rows
            if row[2].get("split") == "test"
            and int(row[2].get("cutoff_year") or 0) in TEST_YEARS
        ]
        years = list(TEST_YEARS)
    else:
        selected = [
            row
            for row in rows
            if row[2].get("split") in ("dev", "validation")
        ]
        years = sorted({int(row[2].get("cutoff_year") or 0) for row in selected})
    if args.limit:
        selected = selected[: args.limit]
    client = DeepSeekClient()
    results = []
    for year in years:
        boundary = parse_date(f"{year}-01-01")
        year_test = [row for row in selected if int(row[2].get("cutoff_year") or 0) == year]
        for fold in range(FOLDS):
            train_rows = [
                row
                for row in rows
                if fold_of(company_key(row[2]), args.seed) != fold
                and row[1] < boundary
            ]
            fold_test = [
                row for row in year_test if fold_of(company_key(row[2]), args.seed) == fold
            ]
            if not train_rows or not fold_test:
                continue
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        run_one,
                        client,
                        record,
                        train_rows,
                        by_company,
                        args.k,
                        args.mode,
                        args.max_tokens,
                    ): record
                    for _, _, record in fold_test
                }
                for future in as_completed(futures):
                    results.append(future.result())
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "per_claim.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    valid = sum(1 for row in results if row["parsed"] and row["parsed"].get("label") in LABELS)
    print(f"wrote {len(results)} rows ({valid} valid) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
