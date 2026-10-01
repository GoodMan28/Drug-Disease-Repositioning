# Same as run_all.ps1 but skips the two 5-fold runs (already done) and writes a
# progress file results\logs\progress.txt (one line per finished step).
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".venv\Scripts\python.exe"
$prog = "results\logs\progress.txt"
New-Item -ItemType Directory -Force results\logs | Out-Null

$steps = @(
  @("F_ablation", "scripts/04_ablation.py --dataset F --protocol cv5 --repeats 1"),
  @("C_case",     "scripts/06_case_study.py --dataset C"),
  @("F_case",     "scripts/06_case_study.py --dataset F"),
  @("cross_FC",   "scripts/07_cross_dataset.py --source F --target C"),
  @("cross_CF",   "scripts/07_cross_dataset.py --source C --target F"),
  @("F_sens",     "scripts/05_sensitivity.py --dataset F --seeds 2"),
  @("F_cv10",     "scripts/03_evaluate.py --dataset F --protocol cv10 --repeats 1"),
  @("C_cv10",     "scripts/03_evaluate.py --dataset C --protocol cv10 --repeats 1"),
  @("F_lodo",     "scripts/03_evaluate.py --dataset F --protocol lodo --lodo-subset 100"),
  @("C_lodo",     "scripts/03_evaluate.py --dataset C --protocol lodo --lodo-subset 100"),
  @("figures",    "scripts/08_make_figures.py")
)
foreach ($s in $steps) {
  $name = $s[0]
  Add-Content $prog "START $name $(Get-Date -Format HH:mm)"
  $p = Start-Process -FilePath $py -ArgumentList ("-u " + $s[1]) -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "results\logs\$name.log" -RedirectStandardError "results\logs\$name.err"
  $status = if ($p.ExitCode -eq 0) { "DONE" } else { "FAILED(exit $($p.ExitCode))" }
  Add-Content $prog "$status $name $(Get-Date -Format HH:mm)"
}
Add-Content $prog "ALL_FINISHED $(Get-Date -Format HH:mm)"
