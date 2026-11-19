# Đánh giá chính thức (giống cách chấm): seed 2026, 20 lượt, xuất ket-qua-<màn>.json và <màn>.mp4 ở gốc dự án.
#   .\scripts\evaluate.ps1 -Run runs/1-1/run1                    # tự chọn checkpoint tốt nhất theo evals.csv
#   .\scripts\evaluate.ps1 -Ckpt runs/1-1/run1/checkpoints/step_003500000.pt
#   .\scripts\evaluate.ps1 -Run runs/1-2/transfer -Level 1-2
param(
    [string]$Run = "",
    [string]$Ckpt = "",
    [string]$Level = "",
    [int]$Episodes = 20,
    [int]$Seed = 2026,
    [switch]$NoVideo
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not $Ckpt) {
    if (-not $Run) { Write-Host "Cần -Run hoặc -Ckpt." -ForegroundColor Red; exit 1 }
    $Ckpt = & $py scripts/run_summary.py $Run --best-ckpt
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
if (-not $Level) {
    # runs/<level>/<name>/checkpoints/x.pt -> <level>
    $Level = Split-Path -Leaf (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $Ckpt)))
}
$out = "ket-qua-$Level.json"
$cmd = @("-m", "mario_ppo.evaluate", "--ckpt", $Ckpt, "--level", $Level, "--episodes", $Episodes, "--seed", $Seed, "--out", $out)
if (-not $NoVideo) { $cmd += @("--video", "$Level.mp4") }

Write-Host "== python $($cmd -join ' ')" -ForegroundColor Cyan
& $py @cmd
if ($LASTEXITCODE -eq 0) { Write-Host "Đã ghi $out$(if (-not $NoVideo) { " và $Level.mp4" })" -ForegroundColor Green }
exit $LASTEXITCODE
