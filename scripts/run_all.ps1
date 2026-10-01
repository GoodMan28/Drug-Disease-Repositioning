# Runs every experiment of the Results section, most important first.
# About 4 hours on an RTX 3050. Each step writes to results/ and can be re-run alone.
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1

$py = ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force results\logs | Out-Null

& $py scripts/03_evaluate.py --dataset F --protocol cv5 --repeats 5      *> results\logs\F_cv5.log
& $py scripts/03_evaluate.py --dataset C --protocol cv5 --repeats 5      *> results\logs\C_cv5.log
& $py scripts/04_ablation.py --dataset F --protocol cv5 --repeats 1      *> results\logs\F_ablation.log
& $py scripts/06_case_study.py --dataset C                               *> results\logs\C_case.log
& $py scripts/06_case_study.py --dataset F                               *> results\logs\F_case.log
& $py scripts/07_cross_dataset.py --source F --target C                  *> results\logs\cross_FC.log
& $py scripts/07_cross_dataset.py --source C --target F                  *> results\logs\cross_CF.log
& $py scripts/05_sensitivity.py --dataset F --seeds 2                    *> results\logs\F_sens.log
& $py scripts/03_evaluate.py --dataset F --protocol cv10 --repeats 1     *> results\logs\F_cv10.log
& $py scripts/03_evaluate.py --dataset C --protocol cv10 --repeats 1     *> results\logs\C_cv10.log
& $py scripts/03_evaluate.py --dataset F --protocol lodo --lodo-subset 100 *> results\logs\F_lodo.log
& $py scripts/03_evaluate.py --dataset C --protocol lodo --lodo-subset 100 *> results\logs\C_lodo.log
& $py scripts/08_make_figures.py                                         *> results\logs\figures.log
