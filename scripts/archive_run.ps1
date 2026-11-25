# Lưu trữ một run: chép checkpoint tốt nhất + config + progress/evals + video tương ứng vào results/<màn>/<tên run>/
# và (tùy chọn) nén thành .zip để đẩy lên Google Drive chung của nhóm.
#   .\scripts\archive_run.ps1 -Run runs/1-1/run1
#   .\scripts\archive_run.ps1 -Run runs/1-1/run1 -Zip
# Lưu ý: *.pt bị .gitignore bỏ qua -> checkpoint chỉ nằm trên máy/Drive, CSV và config có thể commit.
param(
    [Parameter(Mandatory = $true)][string]$Run,
    [switch]$Zip
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"

$Run = $Run.TrimEnd('/', '\')
$name = Split-Path -Leaf $Run
$level = Split-Path -Leaf (Split-Path -Parent $Run)
$dest = "results/$level/$name"
New-Item -ItemType Directory -Force $dest | Out-Null

foreach ($f in "config.yaml", "progress.csv", "evals.csv", "train.log") {
    if (Test-Path "$Run/$f") { Copy-Item "$Run/$f" $dest -Force }
}
$ckpt = & $py scripts/run_summary.py $Run --best-ckpt
if ($LASTEXITCODE -eq 0 -and $ckpt) {
    Copy-Item $ckpt $dest -Force
    $video = "$Run/videos/$([IO.Path]::GetFileNameWithoutExtension($ckpt)).mp4"
    if (Test-Path $video) { Copy-Item $video $dest -Force }
}
& $py scripts/run_summary.py $Run | Out-File -Encoding utf8 "$dest/summary.txt"
Get-Content "$dest/summary.txt"

if ($Zip) {
    $zipPath = "results/$level-$name.zip"
    Compress-Archive -Path "$dest/*" -DestinationPath $zipPath -Force
    Write-Host "Đã nén: $zipPath" -ForegroundColor Green
}
Write-Host "Đã lưu trữ vào $dest" -ForegroundColor Green
