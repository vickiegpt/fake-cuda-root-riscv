typedef int cublasStatus_t;

#define CUBLAS_STUB(name) cublasStatus_t name(void) { return 15; }

CUBLAS_STUB(cublasCreate_v2)
CUBLAS_STUB(cublasDestroy_v2)
CUBLAS_STUB(cublasSetMathMode)
CUBLAS_STUB(cublasGemmEx)
