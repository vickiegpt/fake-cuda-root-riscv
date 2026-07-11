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

"$CC" -shared -fPIC -O2 -g -Wall -Wextra -Wno-unused-parameter \
  -I"$ROOT/include" $CFLAGS \
  -Wl,-soname,libnvidia-ml.so.1 \
  -o "$ROOT/lib64/libnvidia-ml.so.1" \
  "$ROOT/nvidia_driver_shim/libnvidia_ml.c" \
  -L"$ROOT/lib64" -Wl,-rpath,"$ROOT/lib64" -l:libcuda_nvidia.so.1
ln -sfn libnvidia-ml.so.1 "$ROOT/lib64/libnvidia-ml.so"

"$CC" -shared -fPIC -O2 -g -Wall -Wextra -Wno-unused-parameter \
  -I"$ROOT/include" $CFLAGS \
  -Wl,-soname,libcudart.so.12 \
  -o "$ROOT/lib64/libcudart_nvidia.so.12" \
  "$ROOT/nvidia_driver_shim/libcudart_nvidia.c" \
  -L"$ROOT/lib64" -Wl,-rpath,"$ROOT/lib64" -l:libcuda_nvidia.so.1 -lpthread
ln -sfn libcudart_nvidia.so.12 "$ROOT/lib64/libcudart.so.12"
ln -sfn libcudart_nvidia.so.12 "$ROOT/lib64/libcudart.so"

"$CC" -shared -fPIC -O2 -g -Wall -Wextra -Wno-unused-parameter \
  -I"$ROOT/include" $CFLAGS \
  -Wl,-soname,libcublas.so.12 \
  -o "$ROOT/lib64/libcublas_nvidia.so.12" \
  "$ROOT/nvidia_driver_shim/libcublas_nvidia.c"
ln -sfn libcublas_nvidia.so.12 "$ROOT/lib64/libcublas.so.12"
ln -sfn libcublas_nvidia.so.12 "$ROOT/lib64/libcublas.so"

"$CC" -O2 -g -Wall -Wextra -I"$ROOT/include" \
  -o "$ROOT/nvidia_driver_shim/build/cuda_probe" \
  "$ROOT/nvidia_driver_shim/cuda_probe.c" \
  -L"$ROOT/lib64" -Wl,-rpath,"$ROOT/lib64" -l:libcuda_nvidia.so.1

"$CC" -O2 -g -Wall -Wextra -I"$ROOT/include" \
  -o "$ROOT/nvidia_driver_shim/build/launch_probe" \
  "$ROOT/nvidia_driver_shim/launch_probe.c" \
  -L"$ROOT/lib64" -Wl,-rpath,"$ROOT/lib64" -l:libcuda_nvidia.so.1

"$CC" -O2 -g -Wall -Wextra -I"$ROOT/include" \
  -o "$ROOT/nvidia_driver_shim/build/api_probe" \
  "$ROOT/nvidia_driver_shim/api_probe.c" \
  -L"$ROOT/lib64" -Wl,-rpath,"$ROOT/lib64" -l:libcuda_nvidia.so.1

"$CC" -O2 -g -Wall -Wextra -I"$ROOT/include" \
  -o "$ROOT/nvidia_driver_shim/build/module_image_probe" \
  "$ROOT/nvidia_driver_shim/module_image_probe.c" \
  -L"$ROOT/lib64" -Wl,-rpath,"$ROOT/lib64" -l:libcuda_nvidia.so.1

"$CC" -O2 -g -Wall -Wextra -I"$ROOT/include" \
  -o "$ROOT/nvidia_driver_shim/build/cubin_launch_probe" \
  "$ROOT/nvidia_driver_shim/cubin_launch_probe.c" \
  -L"$ROOT/lib64" -Wl,-rpath,"$ROOT/lib64" -l:libcuda_nvidia.so.1

"$CC" -O2 -g -Wall -Wextra \
  -o "$ROOT/nvidia_driver_shim/build/rm_probe" \
  "$ROOT/nvidia_driver_shim/rm_probe.c"

"$CC" -O2 -g -Wall -Wextra \
  -o "$ROOT/nvidia_driver_shim/build/channel_probe" \
  "$ROOT/nvidia_driver_shim/channel_probe.c"

"$CC" -O2 -g -Wall -Wextra -I"$ROOT/include" \
  -o "$ROOT/nvidia_driver_shim/build/nvidia-smi" \
  "$ROOT/nvidia_driver_shim/nvidia_smi.c" \
  -L"$ROOT/lib64" -Wl,-rpath,"$ROOT/lib64" -l:libnvidia-ml.so.1 -l:libcuda_nvidia.so.1

echo "built $ROOT/lib64/libcuda_nvidia.so.1"
echo "built $ROOT/lib64/libnvidia-ml.so.1"
echo "built $ROOT/lib64/libcudart_nvidia.so.12"
echo "built $ROOT/lib64/libcublas_nvidia.so.12"
echo "built $ROOT/nvidia_driver_shim/build/cuda_probe"
echo "built $ROOT/nvidia_driver_shim/build/launch_probe"
echo "built $ROOT/nvidia_driver_shim/build/api_probe"
echo "built $ROOT/nvidia_driver_shim/build/module_image_probe"
echo "built $ROOT/nvidia_driver_shim/build/cubin_launch_probe"
echo "built $ROOT/nvidia_driver_shim/build/rm_probe"
echo "built $ROOT/nvidia_driver_shim/build/channel_probe"
echo "built $ROOT/nvidia_driver_shim/build/nvidia-smi"
