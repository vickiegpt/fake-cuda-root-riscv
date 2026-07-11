#include "../include/cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(CUresult result, const char *what)
{
    const char *name = NULL;
    const char *desc = NULL;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &desc);
    fprintf(stderr, "%s failed: %s (%s)\n", what, name ? name : "?", desc ? desc : "?");
    return 1;
}

static int get_attr(CUfunction fn, CUfunction_attribute attr)
{
    int value = -1;
    if (cuFuncGetAttribute(&value, attr, fn) != CUDA_SUCCESS) {
        return -1;
    }
    return value;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <cubin-or-elf> <kernel-name>\n", argv[0]);
        return 64;
    }

    const char *path = argv[1];
    const char *kernel_name = argv[2];
    CUdevice dev = 0;
    CUcontext ctx = NULL;
    CUmodule mod = NULL;
    CUfunction fn = NULL;

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) return fail(result, "cuInit");
    result = cuDeviceGet(&dev, 0);
    if (result != CUDA_SUCCESS) return fail(result, "cuDeviceGet");
    result = cuCtxCreate(&ctx, 0, dev);
    if (result != CUDA_SUCCESS) return fail(result, "cuCtxCreate");
    result = cuModuleLoad(&mod, path);
    if (result != CUDA_SUCCESS) return fail(result, "cuModuleLoad");
    result = cuModuleGetFunction(&fn, mod, kernel_name);
    if (result != CUDA_SUCCESS) return fail(result, "cuModuleGetFunction");

    int regs = get_attr(fn, CU_FUNC_ATTRIBUTE_NUM_REGS);
    int max_threads = get_attr(fn, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    int smem = get_attr(fn, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
    int local = get_attr(fn, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);
    int constant = get_attr(fn, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES);
    printf("cubin_launch_probe loaded path=%s kernel=%s regs=%d max_threads=%d smem=%d local=%d const=%d\n",
           path, kernel_name, regs, max_threads, smem, local, constant);
    if (getenv("LANXIN_NVIDIA_CUDA_REQUIRE_CUBIN_METADATA") != NULL &&
        regs <= 0 && smem <= 0 && local <= 0 && constant <= 0 && max_threads == 1024) {
        fprintf(stderr, "cubin_launch_probe: no parsed cubin metadata for kernel=%s\n",
                kernel_name);
        cuModuleUnload(mod);
        cuCtxDestroy(ctx);
        return 3;
    }

    if (getenv("LANXIN_NVIDIA_CUDA_CUBIN_LAUNCH") != NULL) {
        result = cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL);
        const char *name = NULL;
        cuGetErrorName(result, &name);
        printf("cubin_launch_probe launch_result=%s qmd_env=%s strict_env=%s text_stage=%s\n",
               name ? name : "?",
               getenv("LANXIN_NVIDIA_CUDA_QMD_SUBMIT") ? "set" : "unset",
               getenv("LANXIN_NVIDIA_CUDA_STRICT_LAUNCH") ? "set" : "unset",
               getenv("LANXIN_NVIDIA_CUDA_CODE_STAGE_TEXT") ? "set" : "unset");
        if (result != CUDA_SUCCESS &&
            !((getenv("LANXIN_NVIDIA_CUDA_STRICT_LAUNCH") != NULL ||
               getenv("LANXIN_NVIDIA_CUDA_QMD_SUBMIT") != NULL) &&
              result == CUDA_ERROR_NOT_SUPPORTED)) {
            cuModuleUnload(mod);
            cuCtxDestroy(ctx);
            return 2;
        }
    }

    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return 0;
}
