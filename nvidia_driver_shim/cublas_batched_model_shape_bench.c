#include <cuda.h>
#include <cublas_v2.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { BATCH_COUNT = 64 };

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int run_case(cublasHandle_t handle, int m, int n, int k,
                    cudaDataType output_type,
                    cublasComputeType_t compute_type) {
    const size_t half_size = sizeof(uint16_t);
    const size_t output_size =
        output_type == CUDA_R_32F ? sizeof(float) : sizeof(uint16_t);
    const size_t a_bytes = (size_t)k * (size_t)m * half_size;
    const size_t b_bytes = (size_t)k * (size_t)n * half_size;
    const size_t c_bytes = (size_t)m * (size_t)n * output_size;
    const float alpha_f32 = 1.0f;
    const float beta_f32 = 0.0f;
    const uint16_t alpha_f16 = UINT16_C(0x3c00);
    const uint16_t beta_f16 = UINT16_C(0x0000);
    const void *alpha =
        output_type == CUDA_R_32F ? (const void *)&alpha_f32 : &alpha_f16;
    const void *beta =
        output_type == CUDA_R_32F ? (const void *)&beta_f32 : &beta_f16;
    CUdeviceptr a = 0;
    CUdeviceptr b = 0;
    CUdeviceptr c[BATCH_COUNT] = {0};
    CUdeviceptr a_array = 0;
    CUdeviceptr b_array = 0;
    CUdeviceptr c_array = 0;
    CUdeviceptr host_a[BATCH_COUNT];
    CUdeviceptr host_b[BATCH_COUNT];

    if (cuMemAlloc(&a, a_bytes) != CUDA_SUCCESS ||
        cuMemAlloc(&b, b_bytes) != CUDA_SUCCESS ||
        cuMemsetD8(a, 0, a_bytes) != CUDA_SUCCESS ||
        cuMemsetD8(b, 0, b_bytes) != CUDA_SUCCESS) {
        return 2;
    }
    for (int batch = 0; batch < BATCH_COUNT; ++batch) {
        host_a[batch] = a;
        host_b[batch] = b;
        if (cuMemAlloc(&c[batch], c_bytes) != CUDA_SUCCESS ||
            cuMemsetD8(c[batch], 0, c_bytes) != CUDA_SUCCESS) {
            return 3;
        }
    }
    if (cuMemAlloc(&a_array, sizeof(host_a)) != CUDA_SUCCESS ||
        cuMemAlloc(&b_array, sizeof(host_b)) != CUDA_SUCCESS ||
        cuMemAlloc(&c_array, sizeof(c)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(a_array, host_a, sizeof(host_a)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(b_array, host_b, sizeof(host_b)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(c_array, c, sizeof(c)) != CUDA_SUCCESS) {
        return 4;
    }

    double begin = now_ms();
    cublasStatus_t status = cublasGemmBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha,
        (const void *const *)(uintptr_t)a_array, CUDA_R_16F, k,
        (const void *const *)(uintptr_t)b_array, CUDA_R_16F, k, beta,
        (void *const *)(uintptr_t)c_array, output_type, m, BATCH_COUNT,
        compute_type, CUBLAS_GEMM_DEFAULT);
    CUresult sync_status = cuCtxSynchronize();
    double elapsed_ms = now_ms() - begin;
    double gflops =
        2.0 * (double)m * (double)n * (double)k * BATCH_COUNT /
        elapsed_ms / 1.0e6;
    printf("m=%d n=%d k=%d batch=%d output=%s status=%d sync=%d "
           "elapsed_ms=%.3f gflops=%.3f\n",
           m, n, k, BATCH_COUNT,
           output_type == CUDA_R_32F ? "f32" : "f16",
           status, sync_status, elapsed_ms, gflops);
    return status == CUBLAS_STATUS_SUCCESS && sync_status == CUDA_SUCCESS ? 0 : 5;
}

int main(void) {
    CUdevice device;
    CUcontext context = NULL;
    cublasHandle_t handle = NULL;
    if (cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&context, 0, device) != CUDA_SUCCESS ||
        cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }
    int rc = run_case(handle, 256, 32, 576, CUDA_R_32F,
                      CUBLAS_COMPUTE_32F);
    if (rc == 0) {
        rc = run_case(handle, 512, 32, 256, CUDA_R_16F,
                      CUBLAS_COMPUTE_16F);
    }
    return rc;
}
