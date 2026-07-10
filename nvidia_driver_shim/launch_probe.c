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

int main(void)
{
    CUdevice dev = 0;
    CUcontext ctx = NULL;
    CUmodule mod = NULL;
    CUfunction fn = NULL;
    static const unsigned char fake_image[] = {0x7f, 'E', 'L', 'F', 0};

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) return fail(result, "cuInit");
    result = cuDeviceGet(&dev, 0);
    if (result != CUDA_SUCCESS) return fail(result, "cuDeviceGet");
    result = cuCtxCreate(&ctx, 0, dev);
    if (result != CUDA_SUCCESS) return fail(result, "cuCtxCreate");
    result = cuModuleLoadData(&mod, fake_image);
    if (result != CUDA_SUCCESS) return fail(result, "cuModuleLoadData");
    result = cuModuleGetFunction(&fn, mod, "fake_kernel");
    if (result != CUDA_SUCCESS) return fail(result, "cuModuleGetFunction");

    result = cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL);
    const char *name = NULL;
    cuGetErrorName(result, &name);
    printf("launch_result=%s noop_env=%s pb_env=%s qmd_env=%s wait_env=%s strict_env=%s rm_submit_env=%s\n",
           name ? name : "?",
           getenv("LANXIN_NVIDIA_CUDA_NOOP_KERNEL") ? "set" : "unset",
           getenv("LANXIN_NVIDIA_CUDA_PB_SUBMIT") ? "set" : "unset",
           getenv("LANXIN_NVIDIA_CUDA_QMD_SUBMIT") ? "set" : "unset",
           getenv("LANXIN_NVIDIA_CUDA_WAIT_COMPLETION") ? "set" : "unset",
           getenv("LANXIN_NVIDIA_CUDA_STRICT_LAUNCH") ? "set" : "unset",
           getenv("LANXIN_NVIDIA_CUDA_RM_SUBMIT") ? "set" : "unset");

    cuModuleUnload(mod);
    cuCtxDestroy(ctx);

    if (getenv("LANXIN_NVIDIA_CUDA_STRICT_LAUNCH") != NULL ||
        getenv("LANXIN_NVIDIA_CUDA_RM_SUBMIT") != NULL) {
        return result == CUDA_ERROR_NOT_SUPPORTED ? 0 : 2;
    }
    return result == CUDA_SUCCESS ? 0 : 3;
}
