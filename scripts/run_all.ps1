# 一条命令跑完全部自检与对比（PowerShell）
#
#   pwsh -File scripts\run_all.ps1              # 自检 + 冒烟对比
#   pwsh -File scripts\run_all.ps1 -Full        # 自检 + 完整对比（AE 训练较慢）
#   pwsh -File scripts\run_all.ps1 -Full -Train # 再补一个 IF 策略训练
param(
    [switch]$Full,
    [switch]$Train,
    [int]$Episodes = 200
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo
$env:PYTHONPATH = Join-Path $repo "src"
$env:PYTHONIOENCODING = "utf-8"

function Step($title) {
    Write-Host ""
    Write-Host ("=" * 74) -ForegroundColor Cyan
    Write-Host "  $title" -ForegroundColor Cyan
    Write-Host ("=" * 74) -ForegroundColor Cyan
}

Step "0. 环境"
python -c "import sys, numpy, sklearn; print('python', sys.version.split()[0]); print('numpy', numpy.__version__); print('sklearn', sklearn.__version__)"
try { python -c "import torch; print('torch', torch.__version__)" } catch { Write-Host "torch 未安装（AE 类检测器不可用）" -ForegroundColor Yellow }

Step "1. 平台 + 检测器契约自检"
python scripts\verify_pipeline.py

Step "2. 检测器模板自检"
python -m agentenvs.detectors.templates

Step "3. ae_paper 契约自检（对照 INTERFACE.md 第 8 章）"
python -m ae_repro.verify_contract

Step "4. 冒烟测试"
python -m pytest tests -q

Step "5. 统一口径检测器对比"
if ($Full) {
    python scripts\eval_all_detectors.py
} else {
    python scripts\eval_all_detectors.py --smoke --out-dir results\_smoke_eval
}

if ($Train) {
    Step "6. IF 策略训练（MADDPG，$Episodes episodes）"
    python scripts\train.py --data bundle --detector if --episodes $Episodes --out results\if --plots
}

Write-Host ""
Write-Host "全部完成。主表：results\detector_comparison.md" -ForegroundColor Green
