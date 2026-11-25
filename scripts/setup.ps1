# Cài môi trường cho cả nhóm (Windows).  Chạy từ gốc dự án:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Gpu      # máy có GPU NVIDIA
# Yêu cầu: Python 3.10-3.12 (khuyên 3.11) và Microsoft C++ Build Tools (để biên dịch nes-py).
param(
    [string]$Python = "3.11",
    [switch]$Gpu
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Fail($msg) { Write-Host "LỖI: $msg" -ForegroundColor Red; exit 1 }

# 1. Tạo .venv
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "== Tạo .venv bằng Python $Python"
    py "-$Python" -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "không tìm thấy Python $Python (py -0 để xem các bản đã cài)" }
}
$py = ".venv\Scripts\python.exe"
& $py --version

# 2. Cài thư viện
& $py -m pip install --upgrade pip
if ($Gpu) {
    & $py -m pip install torch
} else {
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}
& $py -m pip install -e .
if ($LASTEXITCODE -ne 0) {
    Fail ("cài đặt thất bại. Nếu báo 'Microsoft Visual C++ 14.0 or greater is required', cài Build Tools:`n" +
          "  winget install Microsoft.VisualStudio.2022.BuildTools --override `"--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended`"`n" +
          "rồi mở terminal mới và chạy lại script này.")
}
& $py -m pip install pandas matplotlib   # đọc progress.csv, vẽ biểu đồ

# 3. Kiểm tra nhanh
Write-Host "== Kiểm tra import"
& $py -c "import torch, gymnasium, nes_py, gym_super_mario_bros, mo_gymnasium, numpy; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gymnasium', gymnasium.__version__, '| numpy', numpy.__version__)"
if ($LASTEXITCODE -ne 0) { Fail "import thất bại" }

Write-Host "== pytest (ban đầu nhiều bài báo NotImplementedError là bình thường)"
& $py -m pytest -q --no-header -rN | Select-Object -Last 3
Write-Host "`nXong. Kích hoạt môi trường: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
