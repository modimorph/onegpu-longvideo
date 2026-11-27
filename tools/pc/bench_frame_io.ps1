# Run the frame I/O micro-bench on Windows. Set $Repo to this checkout and $Lib to a folder with triton-windows
# (pip install --target $Lib triton-windows); $Py is a Python with torch + CUDA.
param([string]$Repo = ".", [string]$Lib = ".\pylib", [string]$Py = "python")
$env:PYTHONPATH = "$Repo\src;$Lib"
$env:CC = "$Lib\triton\runtime\tcc\tcc.exe"
$env:CUDA_PATH = "$Lib\triton\backends\nvidia"
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used --format=csv,noheader
$t0 = Get-Date
& $Py "$Repo\tools\bench_frame_io.py" 2>&1
Write-Output "exit=$LASTEXITCODE wall_s=$(((Get-Date)-$t0).TotalSeconds)"
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used --format=csv,noheader
