# Smoke test: pytest + huấn luyện ngắn 10k bước, rồi in SPS. Dùng cho mốc ngày 5.
#   .\scripts\smoke_test.ps1
#   .\scripts\smoke_test.ps1 -NumEnvs 8 -SkipTests
param(
    [long]$Steps = 10000,
    [int]$NumEnvs = 0,
    [switch]$SkipTests
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not $SkipTests) {
    Write-Host "== pytest -q" -ForegroundColor Cyan
    & $py -m pytest -q
    if ($LASTEXITCODE -ne 0) { Write-Host "pytest chưa PASS hết, dừng smoke test." -ForegroundColor Red; exit 1 }
}

if (Test-Path "runs/1-1/smoke_test") { Remove-Item -Recurse -Force "runs/1-1/smoke_test" }
# Đánh giá cuối run rút gọn (1 lượt, 300 bước, không video) để smoke test chạy nhanh.
& "$PSScriptRoot\train.ps1" -Name smoke_test -Steps $Steps -NumEnvs $NumEnvs `
    -Extra eval_episodes=1, eval_max_steps=300, video=false
exit $LASTEXITCODE
