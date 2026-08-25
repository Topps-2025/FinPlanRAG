# External data storage

The GitHub working tree contains source code, tests, protocols, and small examples only. Large public datasets, downloaded filings, model weights, and generated results live outside the repository.

The default location is:

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
