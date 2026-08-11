param([int]$P, [int]$Start, [int]$End)
$dir = 'D:\Engineering\FinPlanRAG\database\preexperiments'
Start-Process -FilePath python `
  -ArgumentList "run_lofin_full_benchmark.py","--phase","full","--start","$Start","--end","$End" `
  -WorkingDirectory $dir `
  -RedirectStandardOutput "$dir\full_p$P.log" `
  -RedirectStandardError "$dir\full_p$P.err.log" `
  -WindowStyle Hidden
Write-Output "launched P$P [$Start..$End)"
