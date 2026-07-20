#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_dir=${1:-"$script_dir/.x86-oracle-build"}
clang=${CLANG:-clang}
ld_lld=${LD_LLD:-ld.lld}

mkdir -p "$output_dir"

common_cflags=(
    --target=x86_64-linux-gnu
    -O2
    -ffreestanding
    -fno-stack-protector
    -fno-builtin
)

"$clang" "${common_cflags[@]}" -fPIC \
    -c "$script_dir/x86_cuda_link_stub.c" \
    -o "$output_dir/x86_cuda_link_stub.o"
"$ld_lld" -m elf_x86_64 -shared -soname libcuda.so.1 \
    -o "$output_dir/libcuda.so.1" "$output_dir/x86_cuda_link_stub.o"
ln -sfn libcuda.so.1 "$output_dir/libcuda.so"

"$clang" "${common_cflags[@]}" -fPIC \
    -c "$script_dir/x86_cublas_link_stub.c" \
    -o "$output_dir/x86_cublas_link_stub.o"
"$ld_lld" -m elf_x86_64 -shared -soname libcublas.so.12 \
    -o "$output_dir/libcublas.so.12" "$output_dir/x86_cublas_link_stub.o"
ln -sfn libcublas.so.12 "$output_dir/libcublas.so"

build_probe()
{
    local name=$1
    shift

    "$clang" "${common_cflags[@]}" -fPIE -mno-red-zone \
        -c "$script_dir/$name.c" -o "$output_dir/$name.o"
    "$ld_lld" -m elf_x86_64 -pie \
        --dynamic-linker /lib64/ld-linux-x86-64.so.2 \
        -e _start --no-as-needed --allow-shlib-undefined \
        -o "$output_dir/$name" "$output_dir/$name.o" \
        -L"$output_dir" "$@"
}

build_probe x86_cuda_kernel_oracle_probe -lcuda
build_probe x86_cublas_tensor_oracle -lcublas -lcuda

file "$output_dir/x86_cuda_kernel_oracle_probe" \
     "$output_dir/x86_cublas_tensor_oracle"
