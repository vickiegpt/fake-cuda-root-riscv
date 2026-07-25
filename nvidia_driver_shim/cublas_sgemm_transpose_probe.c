#include <cuda.h>
#include <cublas_v2.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static float matrix_value(const float *matrix, cublasOperation_t op,
                          int row, int col, int ld)
{
    return op == CUBLAS_OP_N ?
           matrix[row + col * ld] : matrix[col + row * ld];
}

int main(void)
{
    const int m = 5;
    const int n = 7;
    const int k = 3;
    float host_a[32];
    float host_b[32];
    float host_c[35];
    float expected[35];
    CUdevice device;
    CUcontext context = NULL;
    CUdeviceptr device_a = 0;
    CUdeviceptr device_b = 0;
    CUdeviceptr device_c = 0;
    cublasHandle_t handle = NULL;
    float alpha = 1.25f;
    float beta = 0.0f;
    int failures = 0;

    if (cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&context, 0, device) != CUDA_SUCCESS ||
        cuMemAlloc(&device_a, sizeof(host_a)) != CUDA_SUCCESS ||
        cuMemAlloc(&device_b, sizeof(host_b)) != CUDA_SUCCESS ||
        cuMemAlloc(&device_c, sizeof(host_c)) != CUDA_SUCCESS ||
        cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }

    for (int trans_a = 0; trans_a < 2; trans_a++) {
        for (int trans_b = 0; trans_b < 2; trans_b++) {
            cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
            int lda = trans_a ? k : m;
            int ldb = trans_b ? n : k;
            size_t a_count = (size_t)lda * (trans_a ? m : k);
            size_t b_count = (size_t)ldb * (trans_b ? k : n);
            for (size_t i = 0; i < a_count; i++) {
                host_a[i] = (float)((int)(i % 11) - 5) / 7.0f;
            }
            for (size_t i = 0; i < b_count; i++) {
                host_b[i] = (float)((int)(i % 13) - 6) / 9.0f;
            }
            memset(host_c, 0, sizeof(host_c));
            memset(expected, 0, sizeof(expected));
            for (int col = 0; col < n; col++) {
                for (int row = 0; row < m; row++) {
                    float sum = 0.0f;
                    for (int inner = 0; inner < k; inner++) {
                        sum += matrix_value(host_a, op_a, row, inner, lda) *
                               matrix_value(host_b, op_b, inner, col, ldb);
                    }
                    expected[row + col * m] = alpha * sum;
                }
            }
            if (cuMemcpyHtoD(device_a, host_a, a_count * sizeof(float)) != CUDA_SUCCESS ||
                cuMemcpyHtoD(device_b, host_b, b_count * sizeof(float)) != CUDA_SUCCESS ||
                cuMemsetD8(device_c, 0, sizeof(host_c)) != CUDA_SUCCESS ||
                cublasSgemm(handle, op_a, op_b, m, n, k, &alpha,
                            (const float *)(uintptr_t)device_a, lda,
                            (const float *)(uintptr_t)device_b, ldb, &beta,
                            (float *)(uintptr_t)device_c, m) != CUBLAS_STATUS_SUCCESS ||
                cuCtxSynchronize() != CUDA_SUCCESS ||
                cuMemcpyDtoH(host_c, device_c, sizeof(host_c)) != CUDA_SUCCESS) {
                return 1;
            }
            for (size_t i = 0; i < 35; i++) {
                if (fabsf(host_c[i] - expected[i]) > 2.0e-3f) {
                    failures++;
                }
            }
            printf("cublas_gpu_sgemm_transpose trans=%c,%c failures=%d\n",
                   trans_a ? 'T' : 'N', trans_b ? 'T' : 'N', failures);
        }
    }
    printf("cublas_gpu_sgemm_transpose verified=%s failures=%d\n",
           failures == 0 ? "yes" : "no", failures);
    return failures == 0 ? 0 : 2;
}
