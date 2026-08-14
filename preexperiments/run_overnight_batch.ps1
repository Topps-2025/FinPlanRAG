# Overnight batch: Part B dense/hybrid baselines + Part A L2 readers.
# Sequential on the single CPU (each step depends on the machine being free).
# Logs: D:\Engineering\FinPlanRAG\database\preexperiments\results\logs\
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$PY = "C:\Users\Lenovo\AppData\Local\Programs\Python\Python310\python.exe"
$DIR = "C:\Users\Lenovo\Desktop\Paper\FinPlanRAG\preexperiments"
$LOG = "D:\Engineering\FinPlanRAG\database\preexperiments\results\logs"
New-Item -ItemType Directory -Force -Path $LOG | Out-Null

function Run-Step($name, $args) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $out = Join-Path $LOG "$name`_$stamp.log"
    Write-Host "[$stamp] START $name -> $out"
    & $PY "$DIR\run_finglm_full_dense_hybrid.py" @args 2>&1 | Tee-Object -FilePath $out
    $code = $LASTEXITCODE
    Write-Host "[$stamp] END $name exit=$code"
}

# Each step is independent and always runs; failures are visible in logs.
# 1. FinGLM: CNBM25 (~260min) + parity gate + dense (~3-5h) + hybrid + invariants
Run-Step "finglm_dense_hybrid" @("--only", "all")
# 2. LOFin: dense (parity gate inside, ~1-2h) then hybrid
Run-Step "lofin_dense" @("--index", "dense")
Run-Step "lofin_hybrid" @("--index", "hybrid")
# 3. L2 readers (diagnostic; ~4h LOFin + ~7h FinGLM)
Run-Step "lofin_l2" @("--phase", "l2")
Run-Step "finglm_l2" @("--phase", "l2")
Write-Host "BATCH DONE"
