#include "../include/cuda.h"

#include <math.h>
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
    if (argc != 2) {
        fprintf(stderr, "usage: %s <sm120-mul-probe.cubin>\n", argv[0]);
        return 64;
    }
    const uint32_t count = 4096;
    const size_t bytes = (size_t)count * sizeof(float);
    float *a_host = malloc(bytes);
    float *b_host = malloc(bytes);
    float *out_host = calloc(count, sizeof(float));
    if (a_host == NULL || b_host == NULL || out_host == NULL) {
        return 1;
    }
    for (uint32_t i = 0; i < count; i++) {
        a_host[i] = (float)i * 0.25f;
        b_host[i] = 2.0f;
    }

    CUdevice device = 0;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUdeviceptr a = 0;
    CUdeviceptr b = 0;
    CUdeviceptr output = 0;
    if (check(cuInit(0), "cuInit") || check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
        check(cuCtxCreate(&context, 0, device), "cuCtxCreate") ||
        check(cuModuleLoad(&module, argv[1]), "cuModuleLoad") ||
        check(cuModuleGetFunction(&function, module, "sm120_mul_probe"), "cuModuleGetFunction") ||
        check(cuMemAlloc(&a, bytes), "cuMemAlloc(a)") ||
        check(cuMemAlloc(&b, bytes), "cuMemAlloc(b)") ||
        check(cuMemAlloc(&output, bytes), "cuMemAlloc(output)") ||
        check(cuMemcpyHtoD(a, a_host, bytes), "cuMemcpyHtoD(a)") ||
        check(cuMemcpyHtoD(b, b_host, bytes), "cuMemcpyHtoD(b)")) {
        return 1;
    }

    void *params[] = {&a, &b, &output, (void *)&count};
    if (check(cuLaunchKernel(function, (count + 255) / 256, 1, 1,
                             256, 1, 1, 0, NULL, params, NULL), "cuLaunchKernel") ||
        check(cuCtxSynchronize(), "cuCtxSynchronize") ||
        check(cuMemcpyDtoH(out_host, output, bytes), "cuMemcpyDtoH")) {
        return 1;
    }

    uint32_t failures = 0;
    for (uint32_t i = 0; i < count; i++) {
        float expected = a_host[i] * b_host[i];
        if (fabsf(out_host[i] - expected) > 1e-5f) {
            failures++;
        }
    }
    printf("sm120_mul_probe count=%u failures=%u samples="
           "[0]=%.3f [255]=%.3f [256]=%.3f [511]=%.3f [4095]=%.3f verified=%s\n",
           count, failures, out_host[0], out_host[255], out_host[256],
           out_host[511], out_host[count - 1],
           failures == 0 ? "yes" : "no");
    cuMemFree(output);
    cuMemFree(b);
    cuMemFree(a);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    free(out_host);
    free(b_host);
    free(a_host);
    return failures == 0 ? 0 : 2;
}
