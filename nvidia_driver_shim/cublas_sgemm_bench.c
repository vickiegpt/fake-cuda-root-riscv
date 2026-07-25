#include <cuda.h>
#include <cublas_v2.h>

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

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "usage: %s <m> <n> <k> <iterations>\n", argv[0]);
        return 64;
    }
    uint32_t m = (uint32_t)strtoul(argv[1], NULL, 10);
    uint32_t n = (uint32_t)strtoul(argv[2], NULL, 10);
    uint32_t k = (uint32_t)strtoul(argv[3], NULL, 10);
    unsigned long iterations = strtoul(argv[4], NULL, 10);
    if (m == 0 || n == 0 || k == 0 || iterations == 0) {
        return 64;
    }

    size_t a_bytes = (size_t)m * k * sizeof(float);
    size_t b_bytes = (size_t)k * n * sizeof(float);
    size_t c_bytes = (size_t)m * n * sizeof(float);
    float *a = malloc(a_bytes);
    float *b = malloc(b_bytes);
    float *c = calloc((size_t)m * n, sizeof(float));
    float *reference = calloc((size_t)m * n, sizeof(float));
    if (a == NULL || b == NULL || c == NULL || reference == NULL) {
        return 1;
    }
    for (size_t i = 0; i < a_bytes / sizeof(float); i++) {
        a[i] = 0.5f;
    }
    for (size_t i = 0; i < b_bytes / sizeof(float); i++) {
        b[i] = 0.25f;
    }
    for (size_t i = 0; i < c_bytes / sizeof(float); i++) {
        reference[i] = (float)k * 0.125f;
    }

    CUdevice device;
    CUcontext context = NULL;
    CUdeviceptr device_a = 0;
    CUdeviceptr device_b = 0;
    CUdeviceptr device_c = 0;
    cublasHandle_t handle = NULL;
    float alpha = 1.0f;
    float beta = 0.0f;
    if (cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&context, 0, device) != CUDA_SUCCESS ||
        cuMemAlloc(&device_a, a_bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&device_b, b_bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&device_c, c_bytes) != CUDA_SUCCESS ||
        cuMemcpyHtoD(device_a, a, a_bytes) != CUDA_SUCCESS ||
        cuMemcpyHtoD(device_b, b, b_bytes) != CUDA_SUCCESS ||
        cuMemsetD8(device_c, 0, c_bytes) != CUDA_SUCCESS ||
        cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS ||
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)m, (int)n, (int)k, &alpha,
                    (const float *)(uintptr_t)device_a, (int)m,
                    (const float *)(uintptr_t)device_b, (int)k,
                    &beta, (float *)(uintptr_t)device_c, (int)m) !=
            CUBLAS_STATUS_SUCCESS ||
        cuCtxSynchronize() != CUDA_SUCCESS) {
        fprintf(stderr, "cublas_sgemm_bench: setup or warmup failed\n");
        return 1;
    }

    uint64_t start_ns = now_ns();
    for (unsigned long i = 0; i < iterations; i++) {
        if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        (int)m, (int)n, (int)k, &alpha,
                        (const float *)(uintptr_t)device_a, (int)m,
                        (const float *)(uintptr_t)device_b, (int)k,
                        &beta, (float *)(uintptr_t)device_c, (int)m) !=
            CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublas_sgemm_bench: launch failed\n");
            return 1;
        }
    }
    if (cuCtxSynchronize() != CUDA_SUCCESS) {
        return 1;
    }
    uint64_t elapsed_ns = now_ns() - start_ns;
    if (cuMemcpyDtoH(c, device_c, c_bytes) != CUDA_SUCCESS) {
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
    printf("cublas_gpu_sgemm m=%u n=%u k=%u iterations=%lu avg_ms=%.6f "
           "gflops=%.3f mismatches=%zu max_abs=%.8g verified=%s\n",
           m, n, k, iterations, seconds * 1000.0 / iterations,
           gflops, mismatches, max_abs, mismatches == 0 ? "yes" : "no");

    cublasDestroy(handle);
    cuMemFree(device_c);
    cuMemFree(device_b);
    cuMemFree(device_a);
    cuCtxDestroy(context);
    free(reference);
    free(c);
    free(b);
    free(a);
    return mismatches == 0 ? 0 : 2;
}
