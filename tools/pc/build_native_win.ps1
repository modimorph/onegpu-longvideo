# Build the optional Rust core (native/olv_native) on Windows and install it beside a checkout's src.
# Needs: rustup (stable-x86_64-pc-windows-msvc as default), MSVC Build Tools (VC.Tools.x86.x64 + a Windows SDK),
# a Python with maturin ($Maturin), the target Python ($Py). The target venv needs no pip: a helper venv installs.
param([string]$Repo = ".", [string]$Py = "python", [string]$Maturin = "maturin", [string]$HelperPy = "python")
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
$Repo = (Resolve-Path $Repo).Path  # before Push-Location, so relative paths cannot point into native\olv_native
$t0 = Get-Date
Push-Location "$Repo\native\olv_native"
& $Maturin build --release -i $Py -o "$Repo\native\wheels" 2>&1 | Select-Object -Last 2
$rc = $LASTEXITCODE
Pop-Location
if ($rc -ne 0) { "maturin build failed ($rc): nothing installed"; exit $rc }
$whl = Get-ChildItem "$Repo\native\wheels" -Filter *win_amd64.whl | Where-Object { $_.LastWriteTime -ge $t0 } | Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $whl) { "no wheel written by this build: nothing installed"; exit 1 }
& $HelperPy -m pip install -q --no-deps --upgrade --target "$Repo\src" $whl.FullName
if ($LASTEXITCODE -ne 0) { "pip install of $($whl.Name) failed ($LASTEXITCODE)"; exit $LASTEXITCODE }
# pip --upgrade replaces the package folder: re-copy a Linux build (for the WSL WER stage) if one exists
$lin = Get-ChildItem "$Repo\native\wheels" -Filter *manylinux*.whl -ErrorAction SilentlyContinue | Select-Object -Last 1
if ($lin) { $z = "$env:TEMP\olv_lin.zip"; Copy-Item $lin.FullName $z -Force; Expand-Archive $z "$env:TEMP\olv_lin" -Force
  Copy-Item "$env:TEMP\olv_lin\olv_native\*.so" "$Repo\src\olv_native\" -Force }
$env:PYTHONPATH = "$Repo\src"
& $Py -c "import olv_native, onegpu_longvideo.qa.wer as w; print('olv_native', olv_native.olv_native.__file__, 'wired', w._native is not None)"
