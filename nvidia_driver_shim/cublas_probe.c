#include <cuda.h>
#include <cublas_v2.h>

#include <math.h>
#include <stdio.h>

int main(void)
{
    const float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float b[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    const float expected[] = {76.0f, 100.0f, 103.0f, 136.0f};
    float result[4] = {0};
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUdevice device;
    CUcontext context = NULL;
    CUdeviceptr da = 0;
    CUdeviceptr db = 0;
    CUdeviceptr dc = 0;
    cublasHandle_t handle = NULL;
    int failures = 0;

    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&context, 0, device) != CUDA_SUCCESS ||
        cuMemAlloc(&da, sizeof(a)) != CUDA_SUCCESS ||
        cuMemAlloc(&db, sizeof(b)) != CUDA_SUCCESS ||
        cuMemAlloc(&dc, sizeof(result)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(da, a, sizeof(a)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(db, b, sizeof(b)) != CUDA_SUCCESS ||
        cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS ||
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 3, &alpha,
                    (const float *)(uintptr_t)da, 2,
                    (const float *)(uintptr_t)db, 3, &beta,
                    (float *)(uintptr_t)dc, 2) != CUBLAS_STATUS_SUCCESS ||
        cuMemcpyDtoH(result, dc, sizeof(result)) != CUDA_SUCCESS) {
        fprintf(stderr, "cublas_probe: API failure\n");
        failures = 1;
        goto out;
    }

    for (size_t i = 0; i < 4; i++) {
        if (fabsf(result[i] - expected[i]) > 1e-5f) {
            failures++;
        }
    }
    printf("cublas_probe result=[%.1f %.1f %.1f %.1f] failures=%d verified=%s\n",
           result[0], result[1], result[2], result[3], failures,
           failures == 0 ? "yes" : "no");

out:
    if (handle != NULL) cublasDestroy(handle);
    if (dc != 0) cuMemFree(dc);
    if (db != 0) cuMemFree(db);
    if (da != 0) cuMemFree(da);
    if (context != NULL) cuCtxDestroy(context);
    return failures == 0 ? 0 : 1;
}
