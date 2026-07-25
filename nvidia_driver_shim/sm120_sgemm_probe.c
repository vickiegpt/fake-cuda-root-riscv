#include "../include/cuda.h"

#include <math.h>
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

static void reference_sgemm(const float *a, const float *b, float *c,
                            uint32_t m, uint32_t n, uint32_t k,
                            float alpha, float beta)
{
    for (uint32_t col = 0; col < n; col++) {
        for (uint32_t row = 0; row < m; row++) {
            float sum = 0.0f;
            for (uint32_t inner = 0; inner < k; inner++) {
                sum += a[row + inner * m] * b[inner + col * k];
            }
            c[row + col * m] = alpha * sum + beta * c[row + col * m];
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 6) {
        fprintf(stderr, "usage: %s <cubin> <m> <n> <k> <iterations>\n", argv[0]);
        return 64;
    }

    uint32_t m = (uint32_t)strtoul(argv[2], NULL, 10);
    uint32_t n = (uint32_t)strtoul(argv[3], NULL, 10);
    uint32_t k = (uint32_t)strtoul(argv[4], NULL, 10);
    unsigned long iterations = strtoul(argv[5], NULL, 10);
    if (m == 0 || n == 0 || k == 0 || iterations == 0) {
        return 64;
    }

    size_t a_bytes = (size_t)m * k * sizeof(float);
    size_t b_bytes = (size_t)k * n * sizeof(float);
    size_t c_bytes = (size_t)m * n * sizeof(float);
    float *a = malloc(a_bytes);
    float *b = malloc(b_bytes);
    float *c = malloc(c_bytes);
    float *reference = malloc(c_bytes);
    if (a == NULL || b == NULL || c == NULL || reference == NULL) {
        return 1;
    }
    for (size_t i = 0; i < a_bytes / sizeof(float); i++) {
        a[i] = (float)((int)(i % 17) - 8) / 17.0f;
    }
    for (size_t i = 0; i < b_bytes / sizeof(float); i++) {
        b[i] = (float)((int)(i % 13) - 6) / 13.0f;
    }
    for (size_t i = 0; i < c_bytes / sizeof(float); i++) {
        c[i] = (float)(i % 5) / 19.0f;
    }
    memcpy(reference, c, c_bytes);

    float alpha = 1.0f;
    float beta = 0.0f;
    uint32_t transpose_a = 0;
    uint32_t transpose_b = 0;
    reference_sgemm(a, b, reference, m, n, k, alpha, beta);

    CUdevice device = 0;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUdeviceptr device_a = 0;
    CUdeviceptr device_b = 0;
    CUdeviceptr device_c = 0;
    if (check(cuInit(0), "cuInit") ||
        check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
        check(cuCtxCreate(&context, 0, device), "cuCtxCreate") ||
        check(cuModuleLoad(&module, argv[1]), "cuModuleLoad") ||
        check(cuModuleGetFunction(&function, module, "sm120_sgemm"),
              "cuModuleGetFunction") ||
        check(cuMemAlloc(&device_a, a_bytes), "cuMemAlloc(A)") ||
        check(cuMemAlloc(&device_b, b_bytes), "cuMemAlloc(B)") ||
        check(cuMemAlloc(&device_c, c_bytes), "cuMemAlloc(C)") ||
        check(cuMemcpyHtoD(device_a, a, a_bytes), "cuMemcpyHtoD(A)") ||
        check(cuMemcpyHtoD(device_b, b, b_bytes), "cuMemcpyHtoD(B)") ||
        check(cuMemcpyHtoD(device_c, c, c_bytes), "cuMemcpyHtoD(C)")) {
        return 1;
    }

    void *params[] = {
        &device_a, &device_b, &device_c, &device_c,
        &m, &n, &k, &m, &k, &m, &m,
        &transpose_a, &transpose_b, &alpha, &beta
    };
    unsigned int grid_x = (m + 15U) / 16U;
    unsigned int grid_y = (n + 15U) / 16U;
    if (check(cuLaunchKernel(function, grid_x, grid_y, 1, 16, 16, 1,
                             0, NULL, params, NULL), "cuLaunchKernel(warmup)") ||
        check(cuCtxSynchronize(), "cuCtxSynchronize(warmup)")) {
        return 1;
    }

    uint64_t start_ns = now_ns();
    for (unsigned long i = 0; i < iterations; i++) {
        if (check(cuLaunchKernel(function, grid_x, grid_y, 1, 16, 16, 1,
                                 0, NULL, params, NULL), "cuLaunchKernel")) {
            return 1;
        }
    }
    if (check(cuCtxSynchronize(), "cuCtxSynchronize")) {
        return 1;
    }
    uint64_t elapsed_ns = now_ns() - start_ns;
    if (check(cuMemcpyDtoH(c, device_c, c_bytes), "cuMemcpyDtoH(C)")) {
        return 1;
    }

    size_t mismatches = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < c_bytes / sizeof(float); i++) {
        float delta = fabsf(c[i] - reference[i]);
        if (delta > max_abs) {
            max_abs = delta;
        }
        if (delta > 2.0e-3f) {
            mismatches++;
        }
    }

    double seconds = (double)elapsed_ns / 1.0e9;
    double gflops = 2.0 * (double)m * n * k * iterations / seconds / 1.0e9;
    printf("sm120_sgemm m=%u n=%u k=%u iterations=%lu avg_ms=%.6f "
           "gflops=%.3f mismatches=%zu max_abs=%.8g verified=%s\n",
           m, n, k, iterations, seconds * 1000.0 / iterations,
           gflops, mismatches, max_abs, mismatches == 0 ? "yes" : "no");

    cuMemFree(device_c);
    cuMemFree(device_b);
    cuMemFree(device_a);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    free(reference);
    free(c);
    free(b);
    free(a);
    return mismatches == 0 ? 0 : 2;
}
