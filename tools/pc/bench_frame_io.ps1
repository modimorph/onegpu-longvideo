$env:PYTHONPATH = "C:\cluster\native-gpu\src;C:\cluster\native-gpu\pylib"
$env:CC = "C:\cluster\native-gpu\pylib\triton\runtime\tcc\tcc.exe"
$env:CUDA_PATH = "C:\cluster\native-gpu\pylib\triton\backends\nvidia"
$env:TRITON_CACHE_DIR = "C:\cluster\native-gpu\triton-cache"
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used --format=csv,noheader
$t0 = Get-Date
& C:\cluster\comfyui\.venv\Scripts\python.exe C:\cluster\native-gpu\bench_frame_io.py 2>&1
Write-Output "exit=$LASTEXITCODE wall_s=$(((Get-Date)-$t0).TotalSeconds)"
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used --format=csv,noheader
