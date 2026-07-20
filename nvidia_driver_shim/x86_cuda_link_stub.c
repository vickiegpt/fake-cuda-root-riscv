typedef int CUresult;

#define CUDA_STUB(name) CUresult name(void) { return 999; }

CUDA_STUB(cuInit)
CUDA_STUB(cuDeviceGet)
CUDA_STUB(cuCtxCreate_v2)
CUDA_STUB(cuCtxDestroy_v2)
CUDA_STUB(cuCtxSynchronize)
CUDA_STUB(cuModuleLoad)
CUDA_STUB(cuModuleUnload)
CUDA_STUB(cuModuleGetFunction)
CUDA_STUB(cuMemAlloc_v2)
CUDA_STUB(cuMemFree_v2)
CUDA_STUB(cuMemcpyHtoD_v2)
CUDA_STUB(cuMemcpyDtoH_v2)
CUDA_STUB(cuMemsetD16_v2)
CUDA_STUB(cuLaunchKernel)
CUDA_STUB(cuEventCreate)
CUDA_STUB(cuEventDestroy_v2)
CUDA_STUB(cuEventRecord)
CUDA_STUB(cuEventSynchronize)
CUDA_STUB(cuEventElapsedTime)
