# Revenue-guidance benchmark v4 (released copy)

This directory contains the 1,005-claim benchmark used by the FinPlan paper.
The JSON is a copy of the v4 artifact from the sibling AnchorEvolve workspace,
included here so that the paper's claims, baselines, and decision-layer
results can be audited without a local absolute path.

## Files

- `dataset_v4.json` (1,005 claims): claim id, company/CIK/ticker, sector,
  cutoff date, target period, guidance low/high/mid/width, current-quarter
  value and reliability, claim text, SEC source URL, outcome value/bucket/date,
  outcome source URL and XBRL provenance, verification category, split.
- `manifest_v4.json`: version, split counts, test company list, verification
  counts, and the source manifest.
- `earnings_events.json`: for each claim, the earliest 8-K/8-K/A with SEC Item
  2.02 (Results of Operations and Financial Condition) filed after the cutoff
  and no later than the dataset's 10-Q outcome date. This is the real earnings
  release that publicly reveals the quarter's revenue. Coverage: 993/1,005
  claims; median 2 days before the 10-Q (mean 24, max 160); all 273 test
  claims have an event. Days from cutoff to the earnings event: median 91,
  mean 84. The 12 fallback claims use the 10-Q date.
- `earnings_events_summary.json`: coverage and delay statistics.
- `sec_cache/` (gitignored): raw SEC submissions JSON, regenerable with
  `scripts/build_earnings_events.py`.

SHA-256:

```text
dataset_v4.json   cf24c311806cc15d204b19b8a52c8f3bae6e91bbee563f448cf6643d40923f24
manifest_v4.json  54d8db4226c25ffd8cb51d23fa24a78ff9a77f98639abb6afdac64aec97290b5
```

## Task

Given a management revenue range for the next fiscal quarter and evidence
available at the 8-K cutoff, predict whether reported revenue will land
`below`, `within`, or `above` the range. The outcome is the next reported
quarter, not a static answer in the filing.

## Construction

Two retrieval paths feed the same extractor:

1. a company list of 200 large U.S. filers, with 8-K Item 2.02 filings
   sampled across 2010--2025 and paired with the next earnings release;
2. EDGAR full-text search for revenue-guidance phrases such as "revenue in the
   range of", "revenue guidance", "revenue is expected to be", and "sales in
   the range of".

For each candidate, the builder extracts the next-quarter revenue range from
the earnings release, rejects annual and segment/product/service rows, takes
the realized quarter from XBRL (`RevenueFromContract...`, `Revenues`,
`SalesRevenueNet`, `SalesRevenueGoodsNet`), chooses the first reported fact
near the target quarter, requires the actual to be within 0.5--2.0 times the
guided midpoint and the outcome to be public 20--200 days after the cutoff,
and deduplicates to the earliest claim per company and target period. No human
labels are used; the label is a deterministic rule over the guided range and
the XBRL actual. Every claim is marked `paraphrase`.

The construction scripts are in the sibling AnchorEvolve workspace
(`real_data_v4/build_guidance_v4.py`, `build_search_guidance.py`,
`verify_xbrl_v4.py`, `build_dataset_v4.py`, `fairness_audit_v4.py`). The
released JSON retains the SEC accession/URL for every claim, so the source
filings can be re-checked independently.

## Splits

| split | claims | period |
|---|---:|---|
| dev | 529 | 2010--2018 |
| validation | 203 | 2019--2021 |
| test | 273 | 2022--2025 |
| total | 1,005 | 2010--2025 |

The test split has 126 above, 110 within, and 37 below claims. The year-wise
company-disjoint protocol in the paper re-splits the test claims by company
fold; for test year \(Y\), training uses other company folds with outcome
dates before January 1 of \(Y\).

## Outcome date and earnings release

The dataset's `outcome.date` is the 10-Q filing date. For 993 of 1,005 claims,
the same quarter's revenue was already public in an earlier 8-K Item 2.02
earnings release; the median lead is 2 days, and the maximum is 160 days. The
main company-disjoint benchmark uses the 10-Q date for the training cutoff
(often called the outcome date), which is conservative: it may exclude a
training claim that was public slightly earlier, but it never leaks a future
outcome. The sequential experiment in
`scripts/run_sequential_evidence.py` uses the earlier 8-K date as the real
terminal evidence, because that is the first time the label is public. This is
the point-in-time-correct treatment for the decision problem.

## Label provenance

| verification | all claims | test claims |
|---|---:|---:|
| `xbrl_search` | 810 | 163 |
| `xbrl_verified` | 120 | 59 |
| `xbrl_override` | 6 | 5 |
| `text_only` | 57 | 41 |
| `text_only_mismatch` | 12 | 5 |

`xbrl_verified` means the XBRL fact agrees with the filing text;
`xbrl_override` means the XBRL value replaces a text value; `xbrl_search`
means the fact was found by searching XBRL tags; `text_only` has no standalone
XBRL quarter; `text_only_mismatch` has a later XBRL value that disagrees with
the first report. The verified-only sensitivity keeps the first three
categories (227 of 273 test claims).

## License and provenance

The source filings are public SEC records. The derived claim text is a short
paraphrase/excerpt; the source URL is retained for attribution. The released
JSON is provided for research use under the repository's MIT license; consult
the SEC's terms for the underlying filings.

## Known limitations

- Automatic extraction; quotes are paraphrases until a human verification pass.
- Revenue guidance only; no EPS, capex, or production timing.
- The natural test distribution is above-heavy; balanced accuracy and macro F1
  are the primary metrics.
- 57 claims have no standalone XBRL quarter; 12 have a text/XBRL mismatch.
- The test set is not a new random sample; the rolling-origin benchmark reuses
  the 273 test claims plus 464 earlier claims.

## Reproduction commands

The paper's runners accept `--data`. From the repository root:

```bash
PYTHONPATH="src;scripts" python scripts/run_company_disjoint_benchmark.py \
  --data paper/data/dataset_v4.json

PYTHONPATH="src;scripts" python scripts/run_company_disjoint_benchmark.py \
  --data paper/data/dataset_v4.json --verified-only \
  --out paper/results/company_disjoint_benchmark_verified

PYTHONPATH="src;scripts" python scripts/run_backbone_conformal_diagnostics.py \
  --data paper/data/dataset_v4.json --seeds 1 --models rf --rf-trees 200 \
  --configs base --thresholds 31 --out paper/results/rf_all_s1

PYTHONPATH="src;scripts" python scripts/run_named_baselines.py \
  --data paper/data/dataset_v4.json --verified-only \
  --variants findver,finfact,selfrag,crag,flare,adaptive \
  --workers 8 --out paper/results/named_baselines
```

The named-baseline run requires a DeepSeek endpoint in `ANTHROPIC_BASE_URL`,
`ANTHROPIC_MODEL`, and `ANTHROPIC_AUTH_TOKEN`. The released per-claim outputs
in `paper/results/named_baselines/` allow the metrics to be recomputed without
the API.
