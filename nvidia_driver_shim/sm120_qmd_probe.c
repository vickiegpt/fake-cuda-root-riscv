#include "../include/cuda.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static unsigned int env_dimension(const char *name, unsigned int fallback)
{
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 0);
    if (end == value || *end != '\0' || parsed == 0 || parsed > UINT32_MAX) {
        return fallback;
    }
    return (unsigned int)parsed;
}

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
    bool use_extra = false;
    unsigned long iterations = 1;
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s <sm120-probe.cubin> [--extra] [iterations]\n", argv[0]);
        return 64;
    }
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--extra") == 0) {
            use_extra = true;
            continue;
        }
        char *end = NULL;
        iterations = strtoul(argv[i], &end, 10);
        if (end == argv[i] || *end != '\0' || iterations == 0 || iterations > 10000000UL) {
            fprintf(stderr, "invalid iteration count: %s\n", argv[i]);
            return 64;
        }
    }

    CUdevice device = 0;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUdeviceptr output = 0;
    uint32_t value = 0;

    if (check(cuInit(0), "cuInit") ||
        check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
        check(cuCtxCreate(&context, 0, device), "cuCtxCreate") ||
        check(cuModuleLoad(&module, argv[1]), "cuModuleLoad") ||
        check(cuModuleGetFunction(&function, module, "sm120_store_probe"),
              "cuModuleGetFunction") ||
        check(cuMemAlloc(&output, sizeof(value)), "cuMemAlloc") ||
        check(cuMemcpyHtoD(output, &value, sizeof(value)), "cuMemcpyHtoD")) {
        return 1;
    }

    uint64_t packed_output = output;
    size_t packed_size = sizeof(packed_output);
    void *extra[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, &packed_output,
        CU_LAUNCH_PARAM_BUFFER_SIZE, &packed_size,
        CU_LAUNCH_PARAM_END
    };
    void *kernel_params[] = {&output};
    void **launch_params = use_extra ? NULL : kernel_params;
    void **launch_extra = use_extra ? extra : NULL;
    unsigned int grid_x = env_dimension("LANXIN_PROBE_GRID_X", 1);
    unsigned int grid_y = env_dimension("LANXIN_PROBE_GRID_Y", 1);
    unsigned int grid_z = env_dimension("LANXIN_PROBE_GRID_Z", 1);
    unsigned int block_x = env_dimension("LANXIN_PROBE_BLOCK_X", 1);
    unsigned int block_y = env_dimension("LANXIN_PROBE_BLOCK_Y", 1);
    unsigned int block_z = env_dimension("LANXIN_PROBE_BLOCK_Z", 1);

    if (iterations > 1 &&
        check(cuLaunchKernel(function, grid_x, grid_y, grid_z,
                             block_x, block_y, block_z, 0, NULL,
                             launch_params, launch_extra), "cuLaunchKernel(warmup)")) {
        return 1;
    }
    uint64_t start_ns = now_ns();
    for (unsigned long i = 0; i < iterations; i++) {
        if (check(cuLaunchKernel(function, grid_x, grid_y, grid_z,
                                 block_x, block_y, block_z, 0, NULL,
                                 launch_params, launch_extra), "cuLaunchKernel")) {
            return 1;
        }
    }
    uint64_t elapsed_ns = now_ns() - start_ns;
    if (check(cuCtxSynchronize(), "cuCtxSynchronize") ||
        check(cuMemcpyDtoH(&value, output, sizeof(value)), "cuMemcpyDtoH")) {
        return 1;
    }

    double average_us = (double)elapsed_ns / (double)iterations / 1000.0;
    double launches_per_second = elapsed_ns != 0 ?
                                 (double)iterations * 1000000000.0 / (double)elapsed_ns : 0.0;
    printf("sm120_qmd_probe mode=%s grid=%ux%ux%u block=%ux%ux%u "
           "output=%u expected=42 verified=%s "
           "iterations=%lu avg_launch_us=%.3f launches_per_second=%.1f\n",
           use_extra ? "extra" : "kernelParams",
           grid_x, grid_y, grid_z, block_x, block_y, block_z, value,
           value == 42 ? "yes" : "no", iterations,
           average_us, launches_per_second);

    cuMemFree(output);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    return value == 42 ? 0 : 2;
}
