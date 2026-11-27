# Start a ComfyUI server with the flags this repo expects (configs/comfyui_flags.txt).
# usage: serve_comfy.ps1 -Root C:\path\to\ComfyUI -Port 8191 [-Extra "--reserve-vram 4"]
param([Parameter(Mandatory=$true)][string]$Root, [int]$Port = 8188, [string]$Extra = "")
$flags = (Get-Content (Join-Path $PSScriptRoot "..\..\configs\comfyui_flags.txt") -Raw).Trim()
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$log = Join-Path $env:TEMP "comfy-$Port"
Start-Process -FilePath $py -WorkingDirectory $Root -WindowStyle Hidden `
  -ArgumentList "main.py --listen 127.0.0.1 --port $Port --disable-auto-launch --preview-method none $Extra $flags" `
  -RedirectStandardOutput "$log.log" -RedirectStandardError "$log.err.log"
foreach ($i in 1..180) {
  try { Invoke-RestMethod -TimeoutSec 2 "http://127.0.0.1:$Port/system_stats" | Out-Null; "comfy up on $Port after $i s (flags: $Extra $flags)"; exit 0 } catch { Start-Sleep 1 }
}
"comfy did not come up on $Port"; exit 1
