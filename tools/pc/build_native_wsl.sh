#!/bin/bash
# Build the Rust core in WSL (the WER stage runs there) and install it beside a checkout's src.
# usage: build_native_wsl.sh /mnt/c/path/to/checkout /path/to/target/python
set -e
REPO=$1; PY=$2
[ -x ~/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
[ -x ~/olv-maturin/bin/maturin ] || { python3 -m venv ~/olv-maturin && ~/olv-maturin/bin/pip install -q maturin==1.16.0; }
rm -rf /tmp/olv_native && cp -r "$REPO/native/olv_native" /tmp/olv_native && rm -rf /tmp/olv_native/target
(cd /tmp/olv_native && PATH=~/.cargo/bin:$PATH ~/olv-maturin/bin/maturin build --release --locked -i "$PY" -o "$REPO/native/wheels")
# pip --upgrade replaces the package folder: keep a Windows build (.pyd) that sits there for the native ComfyUI side
SAVE=$(mktemp -d); cp "$REPO"/src/olv_native/*.pyd "$SAVE"/ 2>/dev/null || true
"$PY" -m pip install -q --no-deps --upgrade --target "$REPO/src" "$REPO"/native/wheels/olv_native-*manylinux*.whl
cp "$SAVE"/*.pyd "$REPO"/src/olv_native/ 2>/dev/null || true; rm -rf "$SAVE"
cd "$REPO" && PYTHONPATH=src "$PY" -c "import olv_native, onegpu_longvideo.qa.wer as w; print('olv_native', olv_native.olv_native.__file__, 'wired', w._native is not None)"
