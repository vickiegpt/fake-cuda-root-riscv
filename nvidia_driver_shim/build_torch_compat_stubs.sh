#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cc=${CC:-gcc}

if (($# == 0)); then
  echo "usage: $0 TORCH_SHARED_LIBRARY [...]" >&2
  exit 2
fi

work_dir=$(mktemp -d)
trap 'rm -rf -- "$work_dir"' EXIT
symbols="$work_dir/symbols.txt"
assembly="$work_dir/compat.S"

nm -D --undefined-only "$@" 2>/dev/null \
  | awk '{ symbol=$NF; sub(/@.*/, "", symbol); if (symbol ~ /^(cuda|cublas|cusparse|cufft|cusolver|nvrtc|nvJitLink|_ZN2at4cuda4blas4gemm)/) print symbol }' \
  | LC_ALL=C sort -u >"$symbols"

{
  printf '%s\n' '.text' '.globl fake_cuda_compat_stub' '.type fake_cuda_compat_stub, @function' 'fake_cuda_compat_stub:' '  li a0, 801' '  ret'
  while IFS= read -r symbol; do
    printf '.globl %s\n.type %s, @function\n.set %s, fake_cuda_compat_stub\n' \
      "$symbol" "$symbol" "$symbol"
  done <"$symbols"
} >"$assembly"

"$cc" -shared -fPIC -Wl,-soname,libcuda_compat_stubs.so \
  -o "$root/lib64/libcuda_compat_stubs.so" "$assembly"
printf 'built %s with %s compatibility symbols\n' \
  "$root/lib64/libcuda_compat_stubs.so" "$(wc -l <"$symbols")"
