# Data storage

The GitHub working tree contains source code, tests, protocols, and the
released paper benchmark. Large corpora, downloaded filings, model weights, and
generated result dumps live outside the repository.

The paper's 1,005-claim benchmark is released in `paper/data/`:

```text
paper/data/dataset_v4.json    # 1,005 claims, SEC URLs, labels, provenance
paper/data/manifest_v4.json   # split counts, companies, hashes
paper/data/README.md          # construction rules and license
```

The scripts default to `paper/data/dataset_v4.json`. Pass `--data` to use a
different copy.

Large assets use the external storage root:

```text
D:\Engineering\FinPlanRAG\database
```

Override it on another machine with:

```powershell
$env:FINPLANRAG_STORAGE_ROOT = 'E:\YourPath\FinPlanRAG\database'
```

The maintained storage layout is:

```text
database/
├── data/       # source documents and benchmark inputs
├── models/     # local model weights
├── results/    # generated experiment outputs
├── tmp/        # downloads and intermediate files
└── archive/    # superseded artifacts
```

Do not commit credentials, model weights, raw filings, or generated result dumps. Dataset provenance, licenses, download failures, and checksums should be recorded in a small JSON protocol file next to the experiment that uses them.

## Official baseline source checkouts

The six official baseline repositories are external checkouts, not vendored
into Git:

```text
/root/autodl-tmp/FinPlanRAG/vendor/official_baselines/
```

They are fetched at the pinned commits by
`scripts/clone_official_baselines.sh`; the remote data-disk copy is about
1.7 GB. Set `FINPLAN_OFFICIAL_ROOT` to the checkout root when running
`scripts/run_official_baselines.py`. The corresponding per-claim outputs,
metric suites, comparisons, and `provenance.json` files are kept in
`paper/results/official/{deepseek,glm,qwen3}/` and are small enough for Git.
