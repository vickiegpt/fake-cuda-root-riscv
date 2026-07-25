#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t lanxin_CUdeviceptr;

extern int cuMemcpyDtoH_v2(void *dstHost, lanxin_CUdeviceptr srcDevice, size_t ByteCount);
extern int cuMemcpyHtoD_v2(lanxin_CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);

typedef void *cudaStream_t;
typedef int cudaDataType;

typedef void *cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasOperation_t;
typedef int cublasComputeType_t;
typedef int cublasGemmAlgo_t;
typedef int cublasMath_t;
typedef int cublasSideMode_t;
typedef int cublasFillMode_t;
typedef int cublasDiagType_t;
typedef int cublasPointerMode_t;
typedef void *cublasLtHandle_t;
typedef void *cublasLtMatmulDesc_t;
typedef void *cublasLtMatrixLayout_t;
typedef void *cublasLtMatmulPreference_t;
typedef int cublasLtMatmulDescAttributes_t;
typedef int cublasLtMatrixLayoutAttribute_t;
typedef int cublasLtMatmulPreferenceAttributes_t;
typedef void cublasLtMatmulAlgo_t;
typedef struct {
    int state;
} cublasLtMatmulHeuristicResult_t;

#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_STATUS_ALLOC_FAILED 3
#define CUBLAS_STATUS_INVALID_VALUE 7
#define CUBLAS_STATUS_NOT_SUPPORTED 8

#define LANXIN_CUDA_SUCCESS 0

#define LANXIN_CUDA_R_32F 0
#define LANXIN_CUDA_R_64F 1
#define LANXIN_CUDA_R_16F 2
#define LANXIN_CUDA_R_16BF 14

#define LANXIN_CUBLAS_OP_N 0
#define LANXIN_CUBLAS_OP_T 1
#define LANXIN_CUBLAS_OP_C 2

#define LANXIN_CUBLAS_COMPUTE_32F 68
#define LANXIN_CUBLAS_COMPUTE_64F 70

#define LANXIN_CUBLASLT_MATRIX_LAYOUT_TYPE 0
#define LANXIN_CUBLASLT_MATRIX_LAYOUT_ROWS 2
#define LANXIN_CUBLASLT_MATRIX_LAYOUT_COLS 3
#define LANXIN_CUBLASLT_MATRIX_LAYOUT_LD 4
#define LANXIN_CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT 5
#define LANXIN_CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET 6

#define LANXIN_CUBLASLT_MATMUL_DESC_COMPUTE_TYPE 0
#define LANXIN_CUBLASLT_MATMUL_DESC_SCALE_TYPE 1
#define LANXIN_CUBLASLT_MATMUL_DESC_TRANSA 3
#define LANXIN_CUBLASLT_MATMUL_DESC_TRANSB 4

struct lanxin_cublas_handle {
    uint64_t magic;
    cudaStream_t stream;
    cublasMath_t math;
    cublasPointerMode_t pointer_mode;
};

struct lanxin_cublaslt_matmul_desc {
    uint64_t magic;
    cublasComputeType_t compute_type;
    cudaDataType scale_type;
    cublasOperation_t transa;
    cublasOperation_t transb;
};

struct lanxin_cublaslt_layout {
    uint64_t magic;
    cudaDataType type;
    uint64_t rows;
    uint64_t cols;
    int64_t ld;
    int32_t batch_count;
    int64_t stride;
};

static size_t lanxin_dtype_size(cudaDataType type)
{
    switch (type) {
    case LANXIN_CUDA_R_32F:
        return sizeof(float);
    case LANXIN_CUDA_R_64F:
        return sizeof(double);
    case LANXIN_CUDA_R_16F:
    case LANXIN_CUDA_R_16BF:
        return sizeof(uint16_t);
    default:
        return 0;
    }
}

static float lanxin_half_to_float(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000U) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fU;
    uint32_t mant = (uint32_t)h & 0x03ffU;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400U) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffU;
            bits = sign | ((exp + (127U - 15U)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fU) {
        bits = sign | 0x7f800000U | (mant << 13);
    } else {
        bits = sign | ((exp + (127U - 15U)) << 23) | (mant << 13);
    }

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t lanxin_float_to_half(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000U;
    int32_t exp = (int32_t)((bits >> 23) & 0xffU) - 127 + 15;
    uint32_t mant = bits & 0x7fffffU;

    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;
        }
        mant |= 0x800000U;
        uint32_t shifted = mant >> (uint32_t)(1 - exp + 13);
        uint32_t round = (mant >> (uint32_t)(1 - exp + 12)) & 1U;
        return (uint16_t)(sign | (shifted + round));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00U);
    }
    mant += 0x00001000U;
    if (mant & 0x00800000U) {
        mant = 0;
        exp++;
        if (exp >= 31) {
            return (uint16_t)(sign | 0x7c00U);
        }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static float lanxin_bfloat16_to_float(uint16_t b)
{
    uint32_t bits = (uint32_t)b << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t lanxin_float_to_bfloat16(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1U;
    bits += 0x7fffU + lsb;
    return (uint16_t)(bits >> 16);
}

static double lanxin_read_value(const void *base, cudaDataType type, size_t index)
{
    switch (type) {
    case LANXIN_CUDA_R_32F:
        return (double)((const float *)base)[index];
    case LANXIN_CUDA_R_64F:
        return ((const double *)base)[index];
    case LANXIN_CUDA_R_16F:
        return (double)lanxin_half_to_float(((const uint16_t *)base)[index]);
    case LANXIN_CUDA_R_16BF:
        return (double)lanxin_bfloat16_to_float(((const uint16_t *)base)[index]);
    default:
        return 0.0;
    }
}

static void lanxin_write_value(void *base, cudaDataType type, size_t index, double value)
{
    switch (type) {
    case LANXIN_CUDA_R_32F:
        ((float *)base)[index] = (float)value;
        break;
    case LANXIN_CUDA_R_64F:
        ((double *)base)[index] = value;
        break;
    case LANXIN_CUDA_R_16F:
        ((uint16_t *)base)[index] = lanxin_float_to_half((float)value);
        break;
    case LANXIN_CUDA_R_16BF:
        ((uint16_t *)base)[index] = lanxin_float_to_bfloat16((float)value);
        break;
    default:
        break;
    }
}

static double lanxin_read_scalar(const void *ptr, cublasComputeType_t computeType, cudaDataType fallback_type)
{
    if (ptr == NULL) {
        return 0.0;
    }
    if (computeType == LANXIN_CUBLAS_COMPUTE_64F || fallback_type == LANXIN_CUDA_R_64F) {
        return *(const double *)ptr;
    }
    return (double)*(const float *)ptr;
}

static double lanxin_read_matrix(const void *base, cudaDataType type,
                                 cublasOperation_t op, int row, int col, int ld)
{
    size_t index;
    if (op == LANXIN_CUBLAS_OP_T || op == LANXIN_CUBLAS_OP_C) {
        index = (size_t)col + (size_t)row * (size_t)ld;
    } else {
        index = (size_t)row + (size_t)col * (size_t)ld;
    }
    return lanxin_read_value(base, type, index);
}

static int lanxin_cublas_trace_enabled(void)
{
    static int initialized;
    static int enabled;

    if (!initialized) {
        const char *value = getenv("LANXIN_NVIDIA_CUBLAS_TRACE");
        enabled = value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
        initialized = 1;
    }
    return enabled;
}

static int lanxin_matrix_bytes(cublasOperation_t op, int logical_rows, int logical_cols,
                               int ld, size_t element_size, size_t *bytes)
{
    size_t physical_cols;

    if (logical_rows < 0 || logical_cols < 0 || ld <= 0 || element_size == 0 || bytes == NULL) {
        return -1;
    }
    physical_cols = (op == LANXIN_CUBLAS_OP_T || op == LANXIN_CUBLAS_OP_C)
                        ? (size_t)logical_rows
                        : (size_t)logical_cols;
    if (physical_cols != 0 && (size_t)ld > SIZE_MAX / physical_cols) {
        return -1;
    }
    size_t elements = (size_t)ld * physical_cols;
    if (elements != 0 && element_size > SIZE_MAX / elements) {
        return -1;
    }
    *bytes = elements * element_size;
    return 0;
}

static void *lanxin_stage_from_device(const void *device, size_t bytes)
{
    void *host;

    if (bytes == 0) {
        return calloc(1, 1);
    }
    host = malloc(bytes);
    if (host == NULL) {
        return NULL;
    }
    if (cuMemcpyDtoH_v2(host, (lanxin_CUdeviceptr)(uintptr_t)device, bytes) != LANXIN_CUDA_SUCCESS) {
        free(host);
        return NULL;
    }
    return host;
}

static cublasStatus_t lanxin_gemm_host(cublasOperation_t transa, cublasOperation_t transb,
                                       int m, int n, int k, const void *alpha, const void *A,
                                       cudaDataType Atype, int lda, const void *B, cudaDataType Btype,
                                       int ldb, const void *beta, const void *C, cudaDataType Ctype,
                                       int ldc, void *D, cudaDataType Dtype, int ldd,
                                       cublasComputeType_t computeType)
{
    if (m < 0 || n < 0 || k < 0 || lda <= 0 || ldb <= 0 || ldc <= 0 || ldd <= 0 ||
        A == NULL || B == NULL || D == NULL || lanxin_dtype_size(Atype) == 0 ||
        lanxin_dtype_size(Btype) == 0 || lanxin_dtype_size(Ctype) == 0 ||
        lanxin_dtype_size(Dtype) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    double a = alpha != NULL ? lanxin_read_scalar(alpha, computeType, Dtype) : 1.0;
    double b = beta != NULL ? lanxin_read_scalar(beta, computeType, Dtype) : 0.0;

    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            double acc = 0.0;
            for (int kk = 0; kk < k; kk++) {
                acc += lanxin_read_matrix(A, Atype, transa, row, kk, lda) *
                       lanxin_read_matrix(B, Btype, transb, kk, col, ldb);
            }
            size_t out_index = (size_t)row + (size_t)col * (size_t)ldd;
            double prev = 0.0;
            if (b != 0.0 && C != NULL) {
                prev = lanxin_read_value(C, Ctype, (size_t)row + (size_t)col * (size_t)ldc);
            }
            lanxin_write_value(D, Dtype, out_index, a * acc + b * prev);
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

static cublasStatus_t lanxin_gemm_one(cublasOperation_t transa, cublasOperation_t transb,
                                      int m, int n, int k, const void *alpha, const void *A,
                                      cudaDataType Atype, int lda, const void *B, cudaDataType Btype,
                                      int ldb, const void *beta, const void *C, cudaDataType Ctype,
                                      int ldc, void *D, cudaDataType Dtype, int ldd,
                                      cublasComputeType_t computeType)
{
    size_t asz = lanxin_dtype_size(Atype);
    size_t bsz = lanxin_dtype_size(Btype);
    size_t csz = lanxin_dtype_size(Ctype);
    size_t dsz = lanxin_dtype_size(Dtype);
    size_t a_bytes = 0;
    size_t b_bytes = 0;
    size_t c_bytes = 0;
    size_t d_bytes = 0;
    void *host_a = NULL;
    void *host_b = NULL;
    void *host_c = NULL;
    void *host_d = NULL;
    cublasStatus_t status = CUBLAS_STATUS_ALLOC_FAILED;
    double beta_value;

    if (m < 0 || n < 0 || k < 0 || lda <= 0 || ldb <= 0 || ldc <= 0 || ldd <= 0 ||
        A == NULL || B == NULL || D == NULL || asz == 0 || bsz == 0 || csz == 0 || dsz == 0 ||
        lanxin_matrix_bytes(transa, m, k, lda, asz, &a_bytes) != 0 ||
        lanxin_matrix_bytes(transb, k, n, ldb, bsz, &b_bytes) != 0 ||
        lanxin_matrix_bytes(LANXIN_CUBLAS_OP_N, m, n, ldc, csz, &c_bytes) != 0 ||
        lanxin_matrix_bytes(LANXIN_CUBLAS_OP_N, m, n, ldd, dsz, &d_bytes) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    beta_value = beta != NULL ? lanxin_read_scalar(beta, computeType, Dtype) : 0.0;
    if (lanxin_cublas_trace_enabled()) {
        fprintf(stderr,
                "lanxin-cublas: gemm m=%d n=%d k=%d trans=%d,%d type=%d,%d->%d "
                "ld=%d,%d,%d bytes=%zu,%zu,%zu beta=%g A=%p B=%p D=%p\n",
                m, n, k, transa, transb, Atype, Btype, Dtype, lda, ldb, ldd,
                a_bytes, b_bytes, d_bytes, beta_value, A, B, D);
    }

    host_a = lanxin_stage_from_device(A, a_bytes);
    host_b = lanxin_stage_from_device(B, b_bytes);
    if (host_a == NULL || host_b == NULL) {
        goto out;
    }
    if (beta_value != 0.0 && C != NULL) {
        host_c = lanxin_stage_from_device(C, c_bytes);
        if (host_c == NULL) {
            goto out;
        }
    }
    host_d = calloc(1, d_bytes == 0 ? 1 : d_bytes);
    if (host_d == NULL) {
        goto out;
    }

    status = lanxin_gemm_host(transa, transb, m, n, k, alpha, host_a, Atype, lda,
                              host_b, Btype, ldb, beta, host_c, Ctype, ldc,
                              host_d, Dtype, ldd, computeType);
    if (status == CUBLAS_STATUS_SUCCESS &&
        cuMemcpyHtoD_v2((lanxin_CUdeviceptr)(uintptr_t)D, host_d, d_bytes) != LANXIN_CUDA_SUCCESS) {
        status = CUBLAS_STATUS_INVALID_VALUE;
    }

out:
    free(host_d);
    free(host_c);
    free(host_b);
    free(host_a);
    return status;
}

static cublasStatus_t lanxin_gemm_strided(cublasOperation_t transa, cublasOperation_t transb,
                                          int m, int n, int k, const void *alpha, const void *A,
                                          cudaDataType Atype, int lda, long long strideA,
                                          const void *B, cudaDataType Btype, int ldb, long long strideB,
                                          const void *beta, void *C, cudaDataType Ctype, int ldc,
                                          long long strideC, int batchCount,
                                          cublasComputeType_t computeType)
{
    if (batchCount < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    size_t asz = lanxin_dtype_size(Atype);
    size_t bsz = lanxin_dtype_size(Btype);
    size_t csz = lanxin_dtype_size(Ctype);
    if (asz == 0 || bsz == 0 || csz == 0) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    for (int batch = 0; batch < batchCount; batch++) {
        const void *Ab = (const char *)A + (size_t)batch * (size_t)strideA * asz;
        const void *Bb = (const char *)B + (size_t)batch * (size_t)strideB * bsz;
        void *Cb = (char *)C + (size_t)batch * (size_t)strideC * csz;
        cublasStatus_t st = lanxin_gemm_one(transa, transb, m, n, k, alpha, Ab, Atype, lda,
                                            Bb, Btype, ldb, beta, Cb, Ctype, ldc, Cb, Ctype,
                                            ldc, computeType);
        if (st != CUBLAS_STATUS_SUCCESS) {
            return st;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle)
{
    if (handle == NULL) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    struct lanxin_cublas_handle *h = (struct lanxin_cublas_handle *)calloc(1, sizeof(*h));
    if (h == NULL) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    h->magic = 0x4c584e5643424c41ULL;
    *handle = h;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle)
{
    free(handle);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId)
{
    if (handle != NULL) {
        ((struct lanxin_cublas_handle *)handle)->stream = streamId;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode)
{
    if (handle != NULL) {
        ((struct lanxin_cublas_handle *)handle)->math = mode;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode)
{
    if (mode == NULL) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *mode = handle != NULL ? ((struct lanxin_cublas_handle *)handle)->math : 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode)
{
    if (handle != NULL) {
        ((struct lanxin_cublas_handle *)handle)->pointer_mode = mode;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t *mode)
{
    if (mode == NULL) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *mode = handle != NULL ? ((struct lanxin_cublas_handle *)handle)->pointer_mode : 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes)
{
    (void)handle;
    (void)workspace;
    (void)workspaceSizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

const char *cublasGetStatusString(cublasStatus_t status)
{
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    default:
        return "CUBLAS_STATUS_UNKNOWN";
    }
}

cublasStatus_t cublasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, const char *logFileName)
{
    (void)logIsOn;
    (void)logToStdOut;
    (void)logToStdErr;
    (void)logFileName;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                              int m, int n, int k, const float *alpha, const float *A, int lda,
                              const float *B, int ldb, const float *beta, float *C, int ldc)
{
    (void)handle;
    return lanxin_gemm_one(transa, transb, m, n, k, alpha, A, LANXIN_CUDA_R_32F, lda,
                           B, LANXIN_CUDA_R_32F, ldb, beta, C, LANXIN_CUDA_R_32F,
                           ldc, C, LANXIN_CUDA_R_32F, ldc, LANXIN_CUBLAS_COMPUTE_32F);
}

cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k, const uint16_t *alpha, const uint16_t *A, int lda,
                           const uint16_t *B, int ldb, const uint16_t *beta, uint16_t *C, int ldc)
{
    float alpha_f = alpha != NULL ? lanxin_half_to_float(*alpha) : 1.0f;
    float beta_f = beta != NULL ? lanxin_half_to_float(*beta) : 0.0f;
    (void)handle;
    return lanxin_gemm_one(transa, transb, m, n, k, &alpha_f, A, LANXIN_CUDA_R_16F, lda,
                           B, LANXIN_CUDA_R_16F, ldb, &beta_f, C, LANXIN_CUDA_R_16F,
                           ldc, C, LANXIN_CUDA_R_16F, ldc, LANXIN_CUBLAS_COMPUTE_32F);
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                         cublasOperation_t transb, int m, int n, int k,
                                         const float *alpha, const float *A, int lda, long long strideA,
                                         const float *B, int ldb, long long strideB,
                                         const float *beta, float *C, int ldc, long long strideC,
                                         int batchCount)
{
    (void)handle;
    return lanxin_gemm_strided(transa, transb, m, n, k, alpha, A, LANXIN_CUDA_R_32F,
                               lda, strideA, B, LANXIN_CUDA_R_32F, ldb, strideB, beta,
                               C, LANXIN_CUDA_R_32F, ldc, strideC, batchCount,
                               LANXIN_CUBLAS_COMPUTE_32F);
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k, const void *alpha, const void *A, cudaDataType Atype,
                            int lda, const void *B, cudaDataType Btype, int ldb, const void *beta,
                            void *C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType,
                            cublasGemmAlgo_t algo)
{
    (void)handle;
    (void)algo;
    return lanxin_gemm_one(transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype,
                           ldb, beta, C, Ctype, ldc, C, Ctype, ldc, computeType);
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                          cublasOperation_t transb, int m, int n, int k,
                                          const void *alpha, const void *A, cudaDataType Atype,
                                          int lda, long long int strideA, const void *B,
                                          cudaDataType Btype, int ldb, long long int strideB,
                                          const void *beta, void *C, cudaDataType Ctype, int ldc,
                                          long long int strideC, int batchCount,
                                          cublasComputeType_t computeType, cublasGemmAlgo_t algo)
{
    (void)handle;
    (void)algo;
    return lanxin_gemm_strided(transa, transb, m, n, k, alpha, A, Atype, lda, strideA,
                               B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC,
                               batchCount, computeType);
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                   cublasOperation_t transb, int m, int n, int k,
                                   const void *alpha, const void *const Aarray[], cudaDataType Atype,
                                   int lda, const void *const Barray[], cudaDataType Btype, int ldb,
                                   const void *beta, void *const Carray[], cudaDataType Ctype, int ldc,
                                   int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo)
{
    (void)handle;
    (void)algo;
    if (batchCount < 0 || Aarray == NULL || Barray == NULL || Carray == NULL) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (int batch = 0; batch < batchCount; batch++) {
        cublasStatus_t st = lanxin_gemm_one(transa, transb, m, n, k, alpha, Aarray[batch],
                                            Atype, lda, Barray[batch], Btype, ldb, beta,
                                            Carray[batch], Ctype, ldc, Carray[batch], Ctype,
                                            ldc, computeType);
        if (st != CUBLAS_STATUS_SUCCESS) {
            return st;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                  const float *alpha, const float *const A[], int lda, float *const B[],
                                  int ldb, int batchCount)
{
    (void)handle; (void)side; (void)uplo; (void)trans; (void)diag; (void)m; (void)n;
    (void)alpha; (void)A; (void)lda; (void)B; (void)ldb; (void)batchCount;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemm_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasCgemm_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasZgemm_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSgemmEx() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasDgemmStridedBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasCgemmStridedBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasZgemmStridedBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSgemv_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasDgemv_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasCgemv_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasZgemv_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSdot_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasDdot_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasCdotu_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasCdotc_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasZdotu_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasZdotc_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasDotEx() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSgeqrfBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasDgeqrfBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasCgeqrfBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasZgeqrfBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSgetrfBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasDgetrfBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasCgetrfBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasZgetrfBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSgetrsBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasDgetrsBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasCgetrsBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasZgetrsBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSgelsBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasDgelsBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasCgelsBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasZgelsBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasStrsm_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasDtrsm_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasCtrsm_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasZtrsm_v2() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasDtrsmBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasCtrsmBatched() { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasZtrsmBatched() { return CUBLAS_STATUS_SUCCESS; }

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
                                         cublasComputeType_t computeType,
                                         cudaDataType scaleType)
{
    if (matmulDesc == NULL) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    struct lanxin_cublaslt_matmul_desc *desc = calloc(1, sizeof(*desc));
    if (desc == NULL) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    desc->magic = 0x4c584e4c544d4d44ULL;
    desc->compute_type = computeType;
    desc->scale_type = scaleType;
    desc->transa = LANXIN_CUBLAS_OP_N;
    desc->transb = LANXIN_CUBLAS_OP_N;
    *matmulDesc = desc;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc)
{
    free(matmulDesc);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                              cublasLtMatmulDescAttributes_t attr,
                                              const void *buf,
                                              size_t sizeInBytes)
{
    struct lanxin_cublaslt_matmul_desc *desc = (struct lanxin_cublaslt_matmul_desc *)matmulDesc;
    if (desc == NULL || buf == NULL) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (attr == LANXIN_CUBLASLT_MATMUL_DESC_TRANSA && sizeInBytes >= sizeof(int32_t)) {
        desc->transa = *(const int32_t *)buf;
    } else if (attr == LANXIN_CUBLASLT_MATMUL_DESC_TRANSB && sizeInBytes >= sizeof(int32_t)) {
        desc->transb = *(const int32_t *)buf;
    } else if (attr == LANXIN_CUBLASLT_MATMUL_DESC_COMPUTE_TYPE && sizeInBytes >= sizeof(int32_t)) {
        desc->compute_type = *(const int32_t *)buf;
    } else if (attr == LANXIN_CUBLASLT_MATMUL_DESC_SCALE_TYPE && sizeInBytes >= sizeof(int32_t)) {
        desc->scale_type = *(const int32_t *)buf;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout,
                                          cudaDataType type,
                                          uint64_t rows,
                                          uint64_t cols,
                                          int64_t ld)
{
    if (matLayout == NULL) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    struct lanxin_cublaslt_layout *layout = calloc(1, sizeof(*layout));
    if (layout == NULL) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    layout->magic = 0x4c584e4c544c4159ULL;
    layout->type = type;
    layout->rows = rows;
    layout->cols = cols;
    layout->ld = ld;
    layout->batch_count = 1;
    layout->stride = 0;
    *matLayout = layout;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout)
{
    free(matLayout);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                                cublasLtMatrixLayoutAttribute_t attr,
                                                const void *buf,
                                                size_t sizeInBytes)
{
    struct lanxin_cublaslt_layout *layout = (struct lanxin_cublaslt_layout *)matLayout;
    if (layout == NULL || buf == NULL) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    switch (attr) {
    case LANXIN_CUBLASLT_MATRIX_LAYOUT_TYPE:
        if (sizeInBytes >= sizeof(uint32_t)) layout->type = *(const uint32_t *)buf;
        break;
    case LANXIN_CUBLASLT_MATRIX_LAYOUT_ROWS:
        if (sizeInBytes >= sizeof(uint64_t)) layout->rows = *(const uint64_t *)buf;
        break;
    case LANXIN_CUBLASLT_MATRIX_LAYOUT_COLS:
        if (sizeInBytes >= sizeof(uint64_t)) layout->cols = *(const uint64_t *)buf;
        break;
    case LANXIN_CUBLASLT_MATRIX_LAYOUT_LD:
        if (sizeInBytes >= sizeof(int64_t)) layout->ld = *(const int64_t *)buf;
        break;
    case LANXIN_CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
        if (sizeInBytes >= sizeof(int32_t)) layout->batch_count = *(const int32_t *)buf;
        break;
    case LANXIN_CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
        if (sizeInBytes >= sizeof(int64_t)) layout->stride = *(const int64_t *)buf;
        break;
    default:
        break;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref)
{
    if (pref == NULL) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *pref = calloc(1, 8);
    return *pref != NULL ? CUBLAS_STATUS_SUCCESS : 3;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref)
{
    free(pref);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                                    cublasLtMatmulPreferenceAttributes_t attr,
                                                    const void *buf,
                                                    size_t sizeInBytes)
{
    (void)pref;
    (void)attr;
    (void)buf;
    (void)sizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle,
                                              cublasLtMatmulDesc_t operationDesc,
                                              cublasLtMatrixLayout_t Adesc,
                                              cublasLtMatrixLayout_t Bdesc,
                                              cublasLtMatrixLayout_t Cdesc,
                                              cublasLtMatrixLayout_t Ddesc,
                                              cublasLtMatmulPreference_t preference,
                                              int requestedAlgoCount,
                                              cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                              int *returnAlgoCount)
{
    (void)lightHandle;
    (void)operationDesc;
    (void)Adesc;
    (void)Bdesc;
    (void)Cdesc;
    (void)Ddesc;
    (void)preference;
    (void)requestedAlgoCount;
    if (requestedAlgoCount > 0 && heuristicResultsArray != NULL) {
        memset(&heuristicResultsArray[0], 0, sizeof(heuristicResultsArray[0]));
        heuristicResultsArray[0].state = CUBLAS_STATUS_SUCCESS;
    }
    if (returnAlgoCount != NULL) {
        *returnAlgoCount = requestedAlgoCount > 0 && heuristicResultsArray != NULL ? 1 : 0;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle,
                              cublasLtMatmulDesc_t computeDesc,
                              const void *alpha,
                              const void *A,
                              cublasLtMatrixLayout_t Adesc,
                              const void *B,
                              cublasLtMatrixLayout_t Bdesc,
                              const void *beta,
                              const void *C,
                              cublasLtMatrixLayout_t Cdesc,
                              void *D,
                              cublasLtMatrixLayout_t Ddesc,
                              const cublasLtMatmulAlgo_t *algo,
                              void *workspace,
                              size_t workspaceSizeInBytes,
                              cudaStream_t stream)
{
    (void)lightHandle;
    (void)algo;
    (void)workspace;
    (void)workspaceSizeInBytes;
    (void)stream;
    struct lanxin_cublaslt_matmul_desc *desc = (struct lanxin_cublaslt_matmul_desc *)computeDesc;
    struct lanxin_cublaslt_layout *a = (struct lanxin_cublaslt_layout *)Adesc;
    struct lanxin_cublaslt_layout *b = (struct lanxin_cublaslt_layout *)Bdesc;
    struct lanxin_cublaslt_layout *c = (struct lanxin_cublaslt_layout *)Cdesc;
    struct lanxin_cublaslt_layout *d = (struct lanxin_cublaslt_layout *)Ddesc;
    if (desc == NULL || a == NULL || b == NULL || c == NULL || d == NULL ||
        A == NULL || B == NULL || D == NULL) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    int m = (int)d->rows;
    int n = (int)d->cols;
    int k = desc->transa == LANXIN_CUBLAS_OP_N ? (int)a->cols : (int)a->rows;
    int batch_count = d->batch_count;
    if (a->batch_count > batch_count) batch_count = a->batch_count;
    if (b->batch_count > batch_count) batch_count = b->batch_count;
    if (c->batch_count > batch_count) batch_count = c->batch_count;
    if (batch_count < 1) batch_count = 1;

    size_t asz = lanxin_dtype_size(a->type);
    size_t bsz = lanxin_dtype_size(b->type);
    size_t csz = lanxin_dtype_size(c->type);
    size_t dsz = lanxin_dtype_size(d->type);
    if (asz == 0 || bsz == 0 || csz == 0 || dsz == 0) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    for (int batch = 0; batch < batch_count; batch++) {
        const void *Ab = (const char *)A + (size_t)batch * (size_t)a->stride * asz;
        const void *Bb = (const char *)B + (size_t)batch * (size_t)b->stride * bsz;
        const void *Cb = C != NULL ? (const char *)C + (size_t)batch * (size_t)c->stride * csz : NULL;
        void *Db = (char *)D + (size_t)batch * (size_t)d->stride * dsz;
        cublasStatus_t st = lanxin_gemm_one(desc->transa, desc->transb, m, n, k, alpha,
                                            Ab, a->type, (int)a->ld, Bb, b->type,
                                            (int)b->ld, beta, Cb, c->type, (int)c->ld,
                                            Db, d->type, (int)d->ld, desc->compute_type);
        if (st != CUBLAS_STATUS_SUCCESS) {
            return st;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

static cublasOperation_t lanxin_op_from_char(char op)
{
    return (op == 't' || op == 'T' || op == 'c' || op == 'C') ? LANXIN_CUBLAS_OP_T : LANXIN_CUBLAS_OP_N;
}

void lanxin_at_cuda_bf16_gemm_stub(char transa, char transb, long m, long n, long k,
                                   float alpha, const uint16_t *A, long lda,
                                   const uint16_t *B, long ldb, float beta,
                                   uint16_t *C, long ldc)
    __asm__("_ZN2at4cuda4blas4gemmIN3c108BFloat16ES4_TnPNSt9enable_ifIXntaaoosr3std7is_sameIT_NS3_4HalfEEE5valuesr3std7is_sameIS6_S4_EE5valuesr3std7is_sameIT0_fEE5valueES6_E4typeELPS4_0EEEvcclllNS_10OpMathTypeIS6_E4typeEPKS6_lSH_lSF_PS8_l");

void lanxin_at_cuda_bf16_gemm_stub(char transa, char transb, long m, long n, long k,
                                   float alpha, const uint16_t *A, long lda,
                                   const uint16_t *B, long ldb, float beta,
                                   uint16_t *C, long ldc)
{
    (void)lanxin_gemm_one(lanxin_op_from_char(transa), lanxin_op_from_char(transb),
                          (int)m, (int)n, (int)k, &alpha, A, LANXIN_CUDA_R_16BF, (int)lda,
                          B, LANXIN_CUDA_R_16BF, (int)ldb, &beta, C, LANXIN_CUDA_R_16BF,
                          (int)ldc, C, LANXIN_CUDA_R_16BF, (int)ldc, LANXIN_CUBLAS_COMPUTE_32F);
}

void lanxin_at_cuda_half_gemm_stub(char transa, char transb, long m, long n, long k,
                                   float alpha, const uint16_t *A, long lda,
                                   const uint16_t *B, long ldb, float beta,
                                   uint16_t *C, long ldc)
    __asm__("_ZN2at4cuda4blas4gemmIN3c104HalfES4_TnPNSt9enable_ifIXntaaoosr3std7is_sameIT_S4_EE5valuesr3std7is_sameIS6_NS3_8BFloat16EEE5valuesr3std7is_sameIT0_fEE5valueES6_E4typeELPS4_0EEEvcclllNS_10OpMathTypeIS6_E4typeEPKS6_lSH_lSF_PS8_l");

void lanxin_at_cuda_half_gemm_stub(char transa, char transb, long m, long n, long k,
                                   float alpha, const uint16_t *A, long lda,
                                   const uint16_t *B, long ldb, float beta,
                                   uint16_t *C, long ldc)
{
    (void)lanxin_gemm_one(lanxin_op_from_char(transa), lanxin_op_from_char(transb),
                          (int)m, (int)n, (int)k, &alpha, A, LANXIN_CUDA_R_16F, (int)lda,
                          B, LANXIN_CUDA_R_16F, (int)ldb, &beta, C, LANXIN_CUDA_R_16F,
                          (int)ldc, C, LANXIN_CUDA_R_16F, (int)ldc, LANXIN_CUBLAS_COMPUTE_32F);
}

void lanxin_at_cuda_double_gemm_stub(char transa, char transb, long m, long n, long k,
                                     double alpha, const double *A, long lda,
                                     const double *B, long ldb, double beta,
                                     double *C, long ldc)
    __asm__("_ZN2at4cuda4blas4gemmIddTnPNSt9enable_ifIXntaaoosr3std7is_sameIT_N3c104HalfEEE5valuesr3std7is_sameIS4_NS5_8BFloat16EEE5valuesr3std7is_sameIT0_fEE5valueES4_E4typeELPd0EEEvcclllNS_10OpMathTypeIS4_E4typeEPKS4_lSH_lSF_PS8_l");

void lanxin_at_cuda_double_gemm_stub(char transa, char transb, long m, long n, long k,
                                     double alpha, const double *A, long lda,
                                     const double *B, long ldb, double beta,
                                     double *C, long ldc)
{
    (void)lanxin_gemm_one(lanxin_op_from_char(transa), lanxin_op_from_char(transb),
                          (int)m, (int)n, (int)k, &alpha, A, LANXIN_CUDA_R_64F, (int)lda,
                          B, LANXIN_CUDA_R_64F, (int)ldb, &beta, C, LANXIN_CUDA_R_64F,
                          (int)ldc, C, LANXIN_CUDA_R_64F, (int)ldc, LANXIN_CUBLAS_COMPUTE_64F);
}

void lanxin_at_cuda_float_gemm_stub(char transa, char transb, long m, long n, long k,
                                    float alpha, const float *A, long lda,
                                    const float *B, long ldb, float beta,
                                    float *C, long ldc)
    __asm__("_ZN2at4cuda4blas4gemmIffTnPNSt9enable_ifIXntaaoosr3std7is_sameIT_N3c104HalfEEE5valuesr3std7is_sameIS4_NS5_8BFloat16EEE5valuesr3std7is_sameIT0_fEE5valueES4_E4typeELPf0EEEvcclllNS_10OpMathTypeIS4_E4typeEPKS4_lSH_lSF_PS8_l");

void lanxin_at_cuda_float_gemm_stub(char transa, char transb, long m, long n, long k,
                                    float alpha, const float *A, long lda,
                                    const float *B, long ldb, float beta,
                                    float *C, long ldc)
{
    (void)lanxin_gemm_one(lanxin_op_from_char(transa), lanxin_op_from_char(transb),
                          (int)m, (int)n, (int)k, &alpha, A, LANXIN_CUDA_R_32F, (int)lda,
                          B, LANXIN_CUDA_R_32F, (int)ldb, &beta, C, LANXIN_CUDA_R_32F,
                          (int)ldc, C, LANXIN_CUDA_R_32F, (int)ldc, LANXIN_CUBLAS_COMPUTE_32F);
}
