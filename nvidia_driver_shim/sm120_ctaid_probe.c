#include "../include/cuda.h"

#include <stdint.h>
#include <stdio.h>

static int check(CUresult result, const char *what)
{
    if (result == CUDA_SUCCESS) {
        return 0;
    }
    const char *name = NULL;
    const char *desc = NULL;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &desc);
    fprintf(stderr, "%s failed: %s (%s)\n", what,
            name != NULL ? name : "?", desc != NULL ? desc : "?");
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <sm120-ctaid-probe.cubin>\n", argv[0]);
        return 64;
    }
    enum { grid_x = 16 };
    uint32_t output_host[grid_x] = {0};
    CUdevice device = 0;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUdeviceptr output = 0;
    if (check(cuInit(0), "cuInit") || check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
        check(cuCtxCreate(&context, 0, device), "cuCtxCreate") ||
        check(cuModuleLoad(&module, argv[1]), "cuModuleLoad") ||
        check(cuModuleGetFunction(&function, module, "sm120_ctaid_probe"), "cuModuleGetFunction") ||
        check(cuMemAlloc(&output, sizeof(output_host)), "cuMemAlloc") ||
        check(cuMemcpyHtoD(output, output_host, sizeof(output_host)), "cuMemcpyHtoD") ||
        check(cuLaunchKernel(function, grid_x, 1, 1, 256, 1, 1,
                             0, NULL, (void *[]){&output}, NULL), "cuLaunchKernel") ||
        check(cuCtxSynchronize(), "cuCtxSynchronize") ||
        check(cuMemcpyDtoH(output_host, output, sizeof(output_host)), "cuMemcpyDtoH")) {
        return 1;
    }
    uint32_t failures = 0;
    for (uint32_t i = 0; i < grid_x; i++) {
        if (output_host[i] != i + 1) {
            failures++;
        }
        printf("%s%u", i == 0 ? "ctaid=[" : ",", output_host[i]);
    }
    printf("] failures=%u verified=%s\n", failures, failures == 0 ? "yes" : "no");
    cuMemFree(output);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    return failures == 0 ? 0 : 2;
}
