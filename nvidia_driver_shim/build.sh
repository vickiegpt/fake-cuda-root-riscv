#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CC="${CC:-gcc}"
CFLAGS="${CFLAGS:-}"

mkdir -p "$ROOT/lib64" "$ROOT/nvidia_driver_shim/build"

"$CC" -shared -fPIC -O2 -g -Wall -Wextra -Wno-unused-parameter \
  -I"$ROOT/include" $CFLAGS \
  -Wl,-soname,libcuda.so.1 \
  -o "$ROOT/lib64/libcuda_nvidia.so.1" \
  "$ROOT/nvidia_driver_shim/libcuda_nvidia.c" \
  -lpthread

"$CC" -O2 -g -Wall -Wextra -I"$ROOT/include" \
  -o "$ROOT/nvidia_driver_shim/build/cuda_probe" \
  "$ROOT/nvidia_driver_shim/cuda_probe.c" \
  -L"$ROOT/lib64" -Wl,-rpath,"$ROOT/lib64" -l:libcuda_nvidia.so.1

"$CC" -O2 -g -Wall -Wextra -I"$ROOT/include" \
  -o "$ROOT/nvidia_driver_shim/build/launch_probe" \
  "$ROOT/nvidia_driver_shim/launch_probe.c" \
  -L"$ROOT/lib64" -Wl,-rpath,"$ROOT/lib64" -l:libcuda_nvidia.so.1

"$CC" -O2 -g -Wall -Wextra \
  -o "$ROOT/nvidia_driver_shim/build/rm_probe" \
  "$ROOT/nvidia_driver_shim/rm_probe.c"

"$CC" -O2 -g -Wall -Wextra \
  -o "$ROOT/nvidia_driver_shim/build/channel_probe" \
  "$ROOT/nvidia_driver_shim/channel_probe.c"

echo "built $ROOT/lib64/libcuda_nvidia.so.1"
echo "built $ROOT/nvidia_driver_shim/build/cuda_probe"
echo "built $ROOT/nvidia_driver_shim/build/launch_probe"
echo "built $ROOT/nvidia_driver_shim/build/rm_probe"
echo "built $ROOT/nvidia_driver_shim/build/channel_probe"
