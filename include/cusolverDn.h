/* Stub cusolverDn.h for hetGPU RISC-V build */
#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* cusolverDnHandle_t;

typedef enum {
  CUSOLVER_STATUS_SUCCESS = 0,
  CUSOLVER_STATUS_NOT_INITIALIZED = 1,
  CUSOLVER_STATUS_ALLOC_FAILED = 2,
  CUSOLVER_STATUS_INVALID_VALUE = 3,
  CUSOLVER_STATUS_ARCH_MISMATCH = 4,
  CUSOLVER_STATUS_MAPPING_ERROR = 5,
  CUSOLVER_STATUS_EXECUTION_FAILED = 6,
  CUSOLVER_STATUS_INTERNAL_ERROR = 7,
  CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
  CUSOLVER_STATUS_NOT_SUPPORTED = 9,
  CUSOLVER_STATUS_ZERO_PIVOT = 10,
  CUSOLVER_STATUS_INVALID_LICENSE = 11,
} cusolverStatus_t;

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle);
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle);
cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId);
cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId);

#ifdef __cplusplus
}
#endif
