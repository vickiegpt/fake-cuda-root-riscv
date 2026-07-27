#include <cuda.h>
#include <cublas_v2.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    MATRIX_DIM = 16,
    MATRIX_ELEMENTS = MATRIX_DIM * MATRIX_DIM,
    BATCH_COUNT = 64,
};

static int check_f32(const float *output, float scale) {
    int failures = 0;
    for (int col = 0; col < MATRIX_DIM; ++col) {
        for (int row = 0; row < MATRIX_DIM; ++row) {
            float expected = col == (row + 1) % MATRIX_DIM ? scale : 0.0f;
            failures +=
                fabsf(output[row + col * MATRIX_DIM] - expected) > 1e-3f;
        }
    }
    return failures;
}

static int check_f16(const uint16_t *output, uint16_t scale) {
    int failures = 0;
    for (int col = 0; col < MATRIX_DIM; ++col) {
        for (int row = 0; row < MATRIX_DIM; ++row) {
            uint16_t expected =
                col == (row + 1) % MATRIX_DIM ? scale : UINT16_C(0x0000);
            failures += output[row + col * MATRIX_DIM] != expected;
        }
    }
    return failures;
}

int main(void) {
    const size_t half_bytes = MATRIX_ELEMENTS * sizeof(uint16_t);
    const size_t float_bytes = MATRIX_ELEMENTS * sizeof(float);
    uint16_t identity[MATRIX_ELEMENTS] = {0};
    uint16_t input[BATCH_COUNT][MATRIX_ELEMENTS];
    float output_f32[BATCH_COUNT][MATRIX_ELEMENTS];
    uint16_t output_f16[BATCH_COUNT][MATRIX_ELEMENTS];
    CUdevice device;
    CUcontext context = NULL;
    CUdeviceptr da[BATCH_COUNT] = {0};
    CUdeviceptr db[BATCH_COUNT] = {0};
    CUdeviceptr dc_f32[BATCH_COUNT] = {0};
    CUdeviceptr dc_f16[BATCH_COUNT] = {0};
    CUdeviceptr da_array = 0;
    CUdeviceptr db_array = 0;
    CUdeviceptr dc_f32_array = 0;
    CUdeviceptr dc_f16_array = 0;
    cublasHandle_t handle = NULL;
    const float alpha_f32 = 1.0f;
    const float beta_f32 = 0.0f;
    const uint16_t alpha_f16 = UINT16_C(0x3c00);
    const uint16_t beta_f16 = UINT16_C(0x0000);
    int failures = 0;

    for (int col = 0; col < MATRIX_DIM; ++col) {
        int row = (col + 1) % MATRIX_DIM;
        identity[row + col * MATRIX_DIM] = UINT16_C(0x3c00);
    }
    memset(input, 0, sizeof(input));
    for (int batch = 0; batch < BATCH_COUNT; ++batch) {
        uint16_t scale =
            (batch & 1) == 0 ? UINT16_C(0x3c00) : UINT16_C(0x4000);
        for (int i = 0; i < MATRIX_DIM; ++i) {
            input[batch][i + i * MATRIX_DIM] = scale;
        }
    }
    memset(output_f32, 0, sizeof(output_f32));
    memset(output_f16, 0, sizeof(output_f16));

    if (cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&context, 0, device) != CUDA_SUCCESS ||
        cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        return 2;
    }
    for (int batch = 0; batch < BATCH_COUNT; ++batch) {
        if (cuMemAlloc(&da[batch], half_bytes) != CUDA_SUCCESS ||
            cuMemAlloc(&db[batch], half_bytes) != CUDA_SUCCESS ||
            cuMemAlloc(&dc_f32[batch], float_bytes) != CUDA_SUCCESS ||
            cuMemAlloc(&dc_f16[batch], half_bytes) != CUDA_SUCCESS ||
            cuMemcpyHtoD(da[batch], identity, half_bytes) != CUDA_SUCCESS ||
            cuMemcpyHtoD(db[batch], input[batch], half_bytes) != CUDA_SUCCESS) {
            return 3;
        }
    }
    if (cuMemAlloc(&da_array, sizeof(da)) != CUDA_SUCCESS ||
        cuMemAlloc(&db_array, sizeof(db)) != CUDA_SUCCESS ||
        cuMemAlloc(&dc_f32_array, sizeof(dc_f32)) != CUDA_SUCCESS ||
        cuMemAlloc(&dc_f16_array, sizeof(dc_f16)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(da_array, da, sizeof(da)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(db_array, db, sizeof(db)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(dc_f32_array, dc_f32, sizeof(dc_f32)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(dc_f16_array, dc_f16, sizeof(dc_f16)) != CUDA_SUCCESS) {
        return 4;
    }

    if (cublasGemmBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            MATRIX_DIM, MATRIX_DIM, MATRIX_DIM,
            &alpha_f32, (const void *const *)(uintptr_t)da_array, CUDA_R_16F,
            MATRIX_DIM, (const void *const *)(uintptr_t)db_array, CUDA_R_16F,
            MATRIX_DIM, &beta_f32, (void *const *)(uintptr_t)dc_f32_array,
            CUDA_R_32F, MATRIX_DIM, BATCH_COUNT, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
        return 5;
    }
    for (int batch = 0; batch < BATCH_COUNT; ++batch) {
        if (cuMemcpyDtoH(output_f32[batch], dc_f32[batch], float_bytes) !=
            CUDA_SUCCESS) {
            return 6;
        }
        failures +=
            check_f32(output_f32[batch], (batch & 1) == 0 ? 1.0f : 2.0f);
    }

    if (cublasGemmBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            MATRIX_DIM, MATRIX_DIM, MATRIX_DIM,
            &alpha_f16, (const void *const *)(uintptr_t)da_array, CUDA_R_16F,
            MATRIX_DIM, (const void *const *)(uintptr_t)db_array, CUDA_R_16F,
            MATRIX_DIM, &beta_f16, (void *const *)(uintptr_t)dc_f16_array,
            CUDA_R_16F, MATRIX_DIM, BATCH_COUNT, CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
        return 7;
    }
    for (int batch = 0; batch < BATCH_COUNT; ++batch) {
        if (cuMemcpyDtoH(output_f16[batch], dc_f16[batch], half_bytes) !=
            CUDA_SUCCESS) {
            return 8;
        }
        failures += check_f16(output_f16[batch],
                              (batch & 1) == 0 ? UINT16_C(0x3c00) :
                                                 UINT16_C(0x4000));
    }

    printf("cublas_batched_device_array failures=%d "
           "f32=[%.1f %.1f] f16=[0x%04x 0x%04x]\n",
           failures, output_f32[0][MATRIX_DIM], output_f32[BATCH_COUNT - 1][MATRIX_DIM],
           output_f16[0][MATRIX_DIM], output_f16[BATCH_COUNT - 1][MATRIX_DIM]);
    return failures ? 9 : 0;
}
