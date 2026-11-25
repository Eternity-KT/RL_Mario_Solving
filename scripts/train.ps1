# Chạy huấn luyện có đặt tên run, tự chọn num_envs theo CPU và ghi log ra file.
#   .\scripts\train.ps1                                   # 1-1, 5M bước, tên run theo thời điểm
#   .\scripts\train.ps1 -Name run1 -Steps 5000000
#   .\scripts\train.ps1 -Name run2 -Extra env.max_no_progress_steps=200,algo.ent_coef=0.02
#   .\scripts\train.ps1 -Level 1-2 -Name transfer -Resume runs/1-1/run1/checkpoints/step_003500000.pt
#   .\scripts\train.ps1 -Name run1 -Continue              # chạy tiếp từ checkpoint mới nhất của run1
# Kết quả: runs/<Level>/<Name>/ (config.yaml, progress.csv, evals.csv, checkpoints/, videos/, train.log)
param(
    [string]$Level = "1-1",
    [string]$Name = (Get-Date -Format "yyyy-MM-dd_HH-mm-ss"),
    [long]$Steps = 0,          # 0 = theo configs/default.yaml
    [int]$NumEnvs = 0,         # 0 = tự chọn theo số luồng CPU
    [int]$TorchThreads = 0,    # 0 = tự chọn (cập nhật PPO trên CPU là phần chậm nhất)
    [int]$Seed = 1,
    [string]$Config = "configs/default.yaml",
    [string]$Resume = "",
    [switch]$Continue,
    [string[]]$Extra = @()     # ghi đè thêm dạng khóa=giá_trị
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "Chưa có .venv, chạy scripts\setup.ps1 trước." -ForegroundColor Red; exit 1 }

$runDir = "runs/$Level/$Name"
if ($Continue) {
    $last = Get-ChildItem "$runDir/checkpoints/step_*.pt" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
    if (-not $last) { Write-Host "Không có checkpoint trong $runDir để chạy tiếp." -ForegroundColor Red; exit 1 }
    $Resume = "$runDir/checkpoints/$($last.Name)"
}

$threads = [Environment]::ProcessorCount
if ($NumEnvs -le 0) {
    if ($threads -ge 16) { $NumEnvs = 16 } elseif ($threads -ge 8) { $NumEnvs = 8 } else { $NumEnvs = 4 }
}
if ($TorchThreads -le 0) {
    # Đo trên máy 22 luồng: 2 luồng -> 37 s/lần cập nhật, 8 luồng -> 14 s, 12 luồng -> 18 s (chậm lại).
    if ($threads -ge 16) { $TorchThreads = 8 } elseif ($threads -ge 8) { $TorchThreads = 4 } else { $TorchThreads = 2 }
}

$args_ = @("-m", "mario_ppo.train", "--config", $Config, "level=$Level", "seed=$Seed", "run_dir=$runDir",
           "algo.num_envs=$NumEnvs", "torch_threads=$TorchThreads")
if ($Steps -gt 0) { $args_ += "algo.total_steps=$Steps" }
if ($Resume) { $args_ += "resume=$Resume" }
$args_ += $Extra

New-Item -ItemType Directory -Force $runDir | Out-Null
$log = "$runDir/train.log"
Write-Host "== $(Get-Date -Format s)  python $($args_ -join ' ')" -ForegroundColor Cyan
"== $(Get-Date -Format s)  python $($args_ -join ' ')" | Add-Content $log
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
& $py @args_ 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE
"== kết thúc lúc $(Get-Date -Format s), mã thoát $code" | Add-Content $log
if (Test-Path "$runDir/progress.csv") { & $py scripts/run_summary.py $runDir }
exit $code
