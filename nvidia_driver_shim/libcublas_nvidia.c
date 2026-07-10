#include <stdint.h>
#include <stdlib.h>

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

#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_STATUS_INVALID_VALUE 7

struct lanxin_cublas_handle {
    uint64_t magic;
    cudaStream_t stream;
    cublasMath_t math;
};

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle)
{
    if (handle == NULL) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    struct lanxin_cublas_handle *h = (struct lanxin_cublas_handle *)calloc(1, sizeof(*h));
    if (h == NULL) {
        return 3;
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

const char *cublasGetStatusString(cublasStatus_t status)
{
    return status == CUBLAS_STATUS_SUCCESS ? "CUBLAS_STATUS_SUCCESS" : "CUBLAS_STATUS_NOT_SUPPORTED";
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
    (void)handle; (void)transa; (void)transb; (void)m; (void)n; (void)k; (void)alpha;
    (void)A; (void)lda; (void)B; (void)ldb; (void)beta; (void)C; (void)ldc;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                         cublasOperation_t transb, int m, int n, int k,
                                         const float *alpha, const float *A, int lda, long long strideA,
                                         const float *B, int ldb, long long strideB,
                                         const float *beta, float *C, int ldc, long long strideC,
                                         int batchCount)
{
    (void)handle; (void)transa; (void)transb; (void)m; (void)n; (void)k; (void)alpha;
    (void)A; (void)lda; (void)strideA; (void)B; (void)ldb; (void)strideB; (void)beta;
    (void)C; (void)ldc; (void)strideC; (void)batchCount;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k, const void *alpha, const void *A, cudaDataType Atype,
                            int lda, const void *B, cudaDataType Btype, int ldb, const void *beta,
                            void *C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType,
                            cublasGemmAlgo_t algo)
{
    (void)handle; (void)transa; (void)transb; (void)m; (void)n; (void)k; (void)alpha;
    (void)A; (void)Atype; (void)lda; (void)B; (void)Btype; (void)ldb; (void)beta;
    (void)C; (void)Ctype; (void)ldc; (void)computeType; (void)algo;
    return CUBLAS_STATUS_SUCCESS;
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
    (void)handle; (void)transa; (void)transb; (void)m; (void)n; (void)k; (void)alpha;
    (void)A; (void)Atype; (void)lda; (void)strideA; (void)B; (void)Btype; (void)ldb;
    (void)strideB; (void)beta; (void)C; (void)Ctype; (void)ldc; (void)strideC;
    (void)batchCount; (void)computeType; (void)algo;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                   cublasOperation_t transb, int m, int n, int k,
                                   const void *alpha, const void *const Aarray[], cudaDataType Atype,
                                   int lda, const void *const Barray[], cudaDataType Btype, int ldb,
                                   const void *beta, void *const Carray[], cudaDataType Ctype, int ldc,
                                   int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo)
{
    (void)handle; (void)transa; (void)transb; (void)m; (void)n; (void)k; (void)alpha;
    (void)Aarray; (void)Atype; (void)lda; (void)Barray; (void)Btype; (void)ldb;
    (void)beta; (void)Carray; (void)Ctype; (void)ldc; (void)batchCount; (void)computeType; (void)algo;
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
