#include "../include/cuda.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s <sm120-local-probe.cubin> [grid-x]\n", argv[0]);
        return 64;
    }
    unsigned long parsed_grid = argc == 3 ? strtoul(argv[2], NULL, 0) : 168;
    if (parsed_grid == 0 || parsed_grid > UINT32_MAX / 256U) {
        fprintf(stderr, "invalid grid-x: %s\n", argc == 3 ? argv[2] : "");
        return 64;
    }
    unsigned int grid = (unsigned int)parsed_grid;
    size_t count = (size_t)grid * 256U;
    size_t output_bytes = count * sizeof(uint32_t);
    uint32_t *output_host = calloc(count, sizeof(*output_host));
    if (output_host == NULL) {
        perror("calloc");
        return 1;
    }
    CUdevice device = 0;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUdeviceptr output = 0;
    if (check(cuInit(0), "cuInit") || check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
        check(cuCtxCreate(&context, 0, device), "cuCtxCreate") ||
        check(cuModuleLoad(&module, argv[1]), "cuModuleLoad") ||
        check(cuModuleGetFunction(&function, module, "sm120_local_probe"), "cuModuleGetFunction") ||
        check(cuMemAlloc(&output, output_bytes), "cuMemAlloc") ||
        check(cuMemcpyHtoD(output, output_host, output_bytes), "cuMemcpyHtoD") ||
        check(cuLaunchKernel(function, grid, 1, 1, 256, 1, 1,
                             0, NULL, (void *[]){&output}, NULL), "cuLaunchKernel") ||
        check(cuCtxSynchronize(), "cuCtxSynchronize") ||
        check(cuMemcpyDtoH(output_host, output, output_bytes), "cuMemcpyDtoH")) {
        free(output_host);
        return 1;
    }
    unsigned int failures = 0;
    for (size_t i = 0; i < count; i++) {
        if (output_host[i] != i + 7) {
            failures++;
        }
    }
    printf("sm120_local_probe grid=%u threads=%zu failures=%u samples=[0]=%u [127]=%u [last]=%u verified=%s\n",
           grid, count, failures, output_host[0], output_host[127], output_host[count - 1],
           failures == 0 ? "yes" : "no");
    cuMemFree(output);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    free(output_host);
    return failures == 0 ? 0 : 2;
}
