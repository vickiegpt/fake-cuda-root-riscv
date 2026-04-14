/*
 * cuda_profiler_api.h - minimal stub for fake_cuda
 * Provides cudaProfilerStart/cudaProfilerStop declarations.
 */
#pragma once
#ifndef CUDA_PROFILER_API_H_
#define CUDA_PROFILER_API_H_

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline cudaError_t cudaProfilerStart(void) {
  return cudaSuccess;
}

static inline cudaError_t cudaProfilerStop(void) {
  return cudaSuccess;
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_PROFILER_API_H_ */
