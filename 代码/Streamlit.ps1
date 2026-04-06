# Streamlit应用启动脚本 (PowerShell版本)
# 用于启动腰椎间盘突出重吸收概率预测计算器

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting Streamlit Application..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Resolve the script directory robustly whether launched directly or via .bat.
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) {
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
}
if (-not $ScriptDir) {
    throw "Unable to determine the script directory."
}
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ScriptDir
Write-Host "Current directory: $ScriptDir" -ForegroundColor Green

$ResultsRoots = @(
    (Join-Path $ProjectRoot "Results"),
    (Join-Path $ScriptDir "Results")
) | Where-Object { Test-Path $_ } | ForEach-Object { (Resolve-Path $_).Path } | Select-Object -Unique

if (-not $ResultsRoots -or $ResultsRoots.Count -eq 0) {
    $ResultsRoots = @((Join-Path $ProjectRoot "Results"))
}

Write-Host "Candidate Results roots:" -ForegroundColor Green
$ResultsRoots | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
Write-Host ""

# 检查模型文件是否存在
# Find model files directly via recursive search (robust with CJK paths)
$ModelFile = $null
$ThresholdFile = $null

foreach ($resultsRoot in $ResultsRoots) {
    $newRoot = Join-Path $resultsRoot "Manuscript_v2"
    if (Test-Path $newRoot) {
        $ModelFile = Get-ChildItem -LiteralPath $newRoot -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "best_model_pipeline_*.pkl" } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        $ThresholdFile = Get-ChildItem -LiteralPath $newRoot -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "*_thresholds*.json" } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($ModelFile) { break }
    }
}

# Fallback: legacy paths
if (-not $ModelFile) {
    foreach ($resultsRoot in $ResultsRoots) {
        if (Test-Path $resultsRoot) {
            $ModelFile = Get-ChildItem -LiteralPath $resultsRoot -Recurse -File -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -like "best_model_pipeline_*.pkl" } |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
            $ThresholdFile = Get-ChildItem -LiteralPath $resultsRoot -Recurse -File -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -like "*_thresholds*.json" } |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($ModelFile) { break }
        }
    }
}

Write-Host "Model search result: $($ModelFile.FullName)" -ForegroundColor Green

if (-not $ModelFile) {
    Write-Host "WARNING: Model file not found!" -ForegroundColor Red
    Write-Host "Expected: <ResultsRoot>\\Manuscript_v2\\run_*\\06_Calculator_Deployment\\exported_model\\best_model_pipeline_*.pkl" -ForegroundColor Yellow
    Write-Host "  or legacy: <ResultsRoot>\\*Machine Learning Modeling*\\deployment\\best_model_pipeline_*.pkl" -ForegroundColor Yellow
    Write-Host "Please run '2_Data_analysis___Model_construction___SHAP_analysis.py' first." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not $ThresholdFile) {
    Write-Host "WARNING: Threshold file not found!" -ForegroundColor Red
    Write-Host "Expected: <ResultsRoot>\\Manuscript_v2\\run_*\\06_Calculator_Deployment\\exported_model\\*_thresholds*.json" -ForegroundColor Yellow
    Write-Host "  or legacy: <ResultsRoot>\\*Machine Learning Modeling*\\deployment\\*_thresholds*.json" -ForegroundColor Yellow
    Write-Host "Please run '2_Data_analysis___Model_construction___SHAP_analysis.py' first." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

$SelectedResultsRoot = $ResultsRoots[0]
foreach ($root in $ResultsRoots) {
    if ($ModelFile.FullName.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        $SelectedResultsRoot = $root
        break
    }
}

Write-Host "Model files found. Starting Streamlit..." -ForegroundColor Green
Write-Host "Using deployment directory: $($ModelFile.Directory.FullName)" -ForegroundColor Green
Write-Host "Using Results root: $SelectedResultsRoot" -ForegroundColor Green
Write-Host ""

# 启动Streamlit应用
$PythonCmd = $null
$PreferredPython = (Join-Path $ProjectRoot ".venv312-gpu\Scripts\python.exe")

if (Test-Path -LiteralPath $PreferredPython) {
    $PythonCmd = $PreferredPython
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCmd = (Get-Command python).Source
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCmd = (Get-Command py).Source
}

if (-not $PythonCmd) {
    Write-Host "ERROR: Python executable not found." -ForegroundColor Red
    Write-Host "Checked: $PreferredPython, C:\Users\lzh71\AppData\Local\Python\bin\python.exe, C:\Users\Lizihang\AppData\Local\Programs\Python\Python311\python.exe, python, py" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Using Python: $PythonCmd" -ForegroundColor Green
$env:LUMBAR_RESULTS_ROOT = $SelectedResultsRoot

try {
    & $PythonCmd -m streamlit run 4_app.py
    $exitCode = $LASTEXITCODE
}
catch {
    Write-Host "" 
    Write-Host "ERROR: Streamlit failed to start." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Application stopped." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($exitCode -ne 0) {
    Write-Host "Streamlit exited with code: $exitCode" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit $exitCode
}