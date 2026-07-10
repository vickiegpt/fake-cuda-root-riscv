#define _GNU_SOURCE
#include "../include/cuda_runtime_api.h"
#include "../include/cuda.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef cudaLaunchKernelExC
#undef cudaLaunchKernelExC
#endif
#ifdef cudaLaunchCooperativeKernel
#undef cudaLaunchCooperativeKernel
#endif
#ifdef cudaStreamBeginCapture
#undef cudaStreamBeginCapture
#endif
#ifdef cudaMemcpyAsync
#undef cudaMemcpyAsync
#endif
#ifdef cudaMemcpy2DAsync
#undef cudaMemcpy2DAsync
#endif
#ifdef cudaGetDeviceProperties
#undef cudaGetDeviceProperties
#endif

#define LANXIN_CUDART_STREAM_MAGIC 0x4c584e5643545351ULL
#define LANXIN_CUDART_EVENT_MAGIC  0x4c584e5643544551ULL

struct lanxin_cudart_stream {
    uint64_t magic;
    CUstream driver;
    unsigned int flags;
};

struct lanxin_cudart_event {
    uint64_t magic;
    CUevent driver;
    unsigned int flags;
};

struct lanxin_cudart_allocation {
    void *ptr;
    size_t bytes;
    int host;
    struct lanxin_cudart_allocation *next;
};

struct lanxin_cudart_module {
    void **handle;
    CUmodule module;
    struct lanxin_cudart_module *next;
};

struct lanxin_cudart_function {
    const void *host;
    CUfunction function;
    char *name;
    struct lanxin_cudart_function *next;
};

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_current_device;
static cudaError_t g_last_error = cudaSuccess;
static struct lanxin_cudart_allocation *g_allocs;
static struct lanxin_cudart_module *g_modules;
static struct lanxin_cudart_function *g_functions;

static __thread dim3 tls_grid = {1, 1, 1};
static __thread dim3 tls_block = {1, 1, 1};
static __thread size_t tls_shared;
static __thread cudaStream_t tls_stream;
static __thread int tls_has_config;

static int trace_enabled(void)
{
    const char *env = getenv("LANXIN_NVIDIA_CUDART_TRACE");
    return env != NULL && env[0] != '\0' && strcmp(env, "0") != 0;
}

#define TRACE(...) do { if (trace_enabled()) fprintf(stderr, "[lanxin cudart] " __VA_ARGS__); } while (0)

static cudaError_t set_last(cudaError_t err)
{
    g_last_error = err;
    return err;
}

static cudaError_t from_cu(CUresult result)
{
    switch (result) {
    case CUDA_SUCCESS: return cudaSuccess;
    case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
    case CUDA_ERROR_INVALID_VALUE: return cudaErrorInvalidValue;
    case CUDA_ERROR_INVALID_DEVICE: return cudaErrorInvalidDevice;
    case CUDA_ERROR_INVALID_HANDLE: return cudaErrorInvalidResourceHandle;
    case CUDA_ERROR_NOT_SUPPORTED: return cudaErrorNotSupported;
    default: return cudaErrorUnknown;
    }
}

static cudaError_t ensure_cuda(void)
{
    return from_cu(cuInit(0));
}

static CUstream driver_stream(cudaStream_t stream)
{
    if (stream == NULL) {
        return NULL;
    }
    if (stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
        return NULL;
    }
    struct lanxin_cudart_stream *s = (struct lanxin_cudart_stream *)stream;
    if (s->magic == LANXIN_CUDART_STREAM_MAGIC) {
        return s->driver;
    }
    return (CUstream)stream;
}

static CUevent driver_event(cudaEvent_t event)
{
    if (event == NULL) {
        return NULL;
    }
    struct lanxin_cudart_event *e = (struct lanxin_cudart_event *)event;
    if (e->magic == LANXIN_CUDART_EVENT_MAGIC) {
        return e->driver;
    }
    return (CUevent)event;
}

static void add_alloc_locked(void *ptr, size_t bytes, int host)
{
    struct lanxin_cudart_allocation *a = (struct lanxin_cudart_allocation *)calloc(1, sizeof(*a));
    if (a == NULL) {
        return;
    }
    a->ptr = ptr;
    a->bytes = bytes;
    a->host = host;
    a->next = g_allocs;
    g_allocs = a;
}

static int remove_alloc_locked(void *ptr, int *host_out)
{
    struct lanxin_cudart_allocation **prev = &g_allocs;
    while (*prev != NULL) {
        struct lanxin_cudart_allocation *a = *prev;
        if (a->ptr == ptr) {
            if (host_out != NULL) {
                *host_out = a->host;
            }
            *prev = a->next;
            free(a);
            return 1;
        }
        prev = &a->next;
    }
    return 0;
}

static int is_device_ptr_locked(const void *ptr)
{
    uintptr_t p = (uintptr_t)ptr;
    for (struct lanxin_cudart_allocation *a = g_allocs; a != NULL; a = a->next) {
        uintptr_t base = (uintptr_t)a->ptr;
        if (!a->host && p >= base && p < base + a->bytes) {
            return 1;
        }
    }
    return 0;
}

static CUfunction lookup_function(const void *host)
{
    pthread_mutex_lock(&g_lock);
    for (struct lanxin_cudart_function *f = g_functions; f != NULL; f = f->next) {
        if (f->host == host) {
            CUfunction out = f->function;
            pthread_mutex_unlock(&g_lock);
            return out;
        }
    }
    pthread_mutex_unlock(&g_lock);
    return NULL;
}

static int query_device_attr_or(CUdevice dev, CUdevice_attribute attr, int fallback)
{
    int value = fallback;
    if (cuDeviceGetAttribute(&value, attr, dev) == CUDA_SUCCESS) {
        return value;
    }
    return fallback;
}

static int runtime_device_attr(CUdevice dev, enum cudaDeviceAttr attr)
{
    switch (attr) {
    case cudaDevAttrMaxThreadsPerBlock:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 1024);
    case cudaDevAttrMaxBlockDimX:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 1024);
    case cudaDevAttrMaxBlockDimY:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, 1024);
    case cudaDevAttrMaxBlockDimZ:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, 64);
    case cudaDevAttrMaxGridDimX:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, 2147483647);
    case cudaDevAttrMaxGridDimY:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, 65535);
    case cudaDevAttrMaxGridDimZ:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, 65535);
    case cudaDevAttrMaxSharedMemoryPerBlock:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, 49152);
    case cudaDevAttrTotalConstantMemory:
        return 65536;
    case cudaDevAttrWarpSize:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_WARP_SIZE, 32);
    case cudaDevAttrMaxPitch:
        return INT32_MAX;
    case cudaDevAttrMaxRegistersPerBlock:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, 65536);
    case cudaDevAttrClockRate:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, 2000000);
    case cudaDevAttrGpuOverlap:
        return 1;
    case cudaDevAttrMultiProcessorCount:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 170);
    case cudaDevAttrCanMapHostMemory:
        return 1;
    case cudaDevAttrComputeMode:
        return cudaComputeModeDefault;
    case cudaDevAttrConcurrentKernels:
        return 1;
    case cudaDevAttrMemoryClockRate:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, 1400000);
    case cudaDevAttrGlobalMemoryBusWidth:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, 512);
    case cudaDevAttrL2CacheSize:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 98304 * 1024);
    case cudaDevAttrMaxThreadsPerMultiProcessor:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, 1536);
    case cudaDevAttrUnifiedAddressing:
        return 1;
    case cudaDevAttrComputeCapabilityMajor:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 12);
    case cudaDevAttrComputeCapabilityMinor:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);
    case cudaDevAttrStreamPrioritiesSupported:
        return 0;
    case cudaDevAttrMaxSharedMemoryPerMultiprocessor:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, 102400);
    case cudaDevAttrMaxRegistersPerMultiprocessor:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, 65536);
    case cudaDevAttrManagedMemory:
        return 0;
    case cudaDevAttrPageableMemoryAccess:
        return 0;
    case cudaDevAttrConcurrentManagedAccess:
        return 0;
    case cudaDevAttrCanUseHostPointerForRegisteredMem:
        return 1;
    case cudaDevAttrCooperativeLaunch:
        return 1;
    case cudaDevAttrCooperativeMultiDeviceLaunch:
        return 0;
    case cudaDevAttrMaxSharedMemoryPerBlockOptin:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, 102400);
    case cudaDevAttrHostRegisterSupported:
        return 1;
    case cudaDevAttrPageableMemoryAccessUsesHostPageTables:
        return 0;
    case cudaDevAttrMaxBlocksPerMultiprocessor:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, 32);
    case cudaDevAttrReservedSharedMemoryPerBlock:
        return query_device_attr_or(dev, CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, 0);
    case cudaDevAttrMemoryPoolsSupported:
        return 0;
    case cudaDevAttrIpcEventSupport:
        return 0;
    default: {
        int value = 0;
        CUresult result = cuDeviceGetAttribute(&value, (CUdevice_attribute)attr, dev);
        return result == CUDA_SUCCESS ? value : 0;
    }
    }
}

cudaError_t CUDARTAPI cudaGetDeviceCount(int *count)
{
    if (count == NULL) {
        return set_last(cudaErrorInvalidValue);
    }
    int n = 0;
    cudaError_t err = from_cu(cuInit(0));
    if (err == cudaSuccess) {
        err = from_cu(cuDeviceGetCount(&n));
    }
    *count = err == cudaSuccess ? n : 0;
    return set_last(err);
}

cudaError_t CUDARTAPI cudaGetDevice(int *device)
{
    if (device == NULL) {
        return set_last(cudaErrorInvalidValue);
    }
    *device = g_current_device;
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaSetDevice(int device)
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return set_last(err);
    }
    if (device < 0 || device >= count) {
        return set_last(cudaErrorInvalidDevice);
    }
    g_current_device = device;
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaSetDeviceFlags(unsigned int flags)
{
    (void)flags;
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaDeviceSynchronize(void)
{
    return set_last(from_cu(cuCtxSynchronize()));
}

cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice)
{
    (void)device;
    (void)peerDevice;
    if (canAccessPeer == NULL) {
        return set_last(cudaErrorInvalidValue);
    }
    *canAccessPeer = 0;
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    (void)peerDevice;
    (void)flags;
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int device)
{
    if (pciBusId == NULL || len <= 0) {
        return set_last(cudaErrorInvalidValue);
    }
    return set_last(from_cu(cuDeviceGetPCIBusId(pciBusId, len, device)));
}

cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
    if (value == NULL) {
        return set_last(cudaErrorInvalidValue);
    }
    CUdevice dev;
    cudaError_t err = from_cu(cuDeviceGet(&dev, device));
    if (err != cudaSuccess) {
        return set_last(err);
    }
    *value = runtime_device_attr(dev, attr);
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaGetDeviceProperties_v2(struct cudaDeviceProp *prop, int device)
{
    if (prop == NULL) {
        return set_last(cudaErrorInvalidValue);
    }
    memset(prop, 0, sizeof(*prop));
    CUdevice dev;
    cudaError_t err = from_cu(cuDeviceGet(&dev, device));
    if (err != cudaSuccess) {
        return set_last(err);
    }
    size_t total = 0;
    (void)cuDeviceGetName(prop->name, sizeof(prop->name), dev);
    (void)cuDeviceTotalMem(&total, dev);
    prop->totalGlobalMem = total;
    (void)cuDeviceGetAttribute(&prop->major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    (void)cuDeviceGetAttribute(&prop->minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    (void)cuDeviceGetAttribute(&prop->multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    (void)cuDeviceGetAttribute(&prop->clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
    (void)cuDeviceGetAttribute(&prop->memoryClockRate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
    (void)cuDeviceGetAttribute(&prop->memoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
    (void)cuDeviceGetAttribute(&prop->l2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);
    prop->sharedMemPerBlock = 49152;
    prop->sharedMemPerBlockOptin = 102400;
    prop->sharedMemPerMultiprocessor = 102400;
    prop->regsPerBlock = 65536;
    prop->regsPerMultiprocessor = 65536;
    prop->warpSize = 32;
    prop->memPitch = (size_t)1 << 31;
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0] = 1024;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->totalConstMem = 65536;
    prop->textureAlignment = 512;
    prop->texturePitchAlignment = 512;
    prop->deviceOverlap = 1;
    prop->asyncEngineCount = 2;
    prop->unifiedAddressing = 1;
    prop->canMapHostMemory = 1;
    prop->concurrentKernels = 1;
    prop->cooperativeLaunch = 1;
    prop->maxThreadsPerMultiProcessor = 1536;
    prop->maxBlocksPerMultiProcessor = 32;
    prop->managedMemory = 0;
    (void)cuDeviceGetUuid((CUuuid *)&prop->uuid, dev);
    char pci[32] = {0};
    if (cuDeviceGetPCIBusId(pci, sizeof(pci), dev) == CUDA_SUCCESS) {
        unsigned int domain = 0, bus = 0, slot = 0;
        (void)sscanf(pci, "%x:%x:%x", &domain, &bus, &slot);
        prop->pciDomainID = (int)domain;
        prop->pciBusID = (int)bus;
        prop->pciDeviceID = (int)slot;
    }
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    return cudaGetDeviceProperties_v2(prop, device);
}

cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
    if (devPtr == NULL) {
        return set_last(cudaErrorInvalidValue);
    }
    CUdeviceptr dptr = 0;
    cudaError_t err = from_cu(cuMemAlloc(&dptr, size));
    if (err == cudaSuccess) {
        *devPtr = (void *)(uintptr_t)dptr;
        pthread_mutex_lock(&g_lock);
        add_alloc_locked(*devPtr, size, 0);
        pthread_mutex_unlock(&g_lock);
    } else {
        *devPtr = NULL;
    }
    return set_last(err);
}

cudaError_t CUDARTAPI cudaMallocManaged(void **devPtr, size_t size, unsigned int flags)
{
    (void)flags;
    return cudaMalloc(devPtr, size);
}

cudaError_t CUDARTAPI cudaFree(void *devPtr)
{
    if (devPtr == NULL) {
        return set_last(cudaSuccess);
    }
    int host = 0;
    pthread_mutex_lock(&g_lock);
    int known = remove_alloc_locked(devPtr, &host);
    pthread_mutex_unlock(&g_lock);
    if (known && host) {
        free(devPtr);
        return set_last(cudaSuccess);
    }
    return set_last(from_cu(cuMemFree((CUdeviceptr)(uintptr_t)devPtr)));
}

cudaError_t CUDARTAPI cudaMemGetInfo(size_t *freeBytes, size_t *totalBytes)
{
    return set_last(from_cu(cuMemGetInfo(freeBytes, totalBytes)));
}

cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
    (void)flags;
    if (pHost == NULL) {
        return set_last(cudaErrorInvalidValue);
    }
    void *ptr = NULL;
    size_t alloc_size = size == 0 ? 1 : size;
    if (posix_memalign(&ptr, 256, alloc_size) != 0 || ptr == NULL) {
        *pHost = NULL;
        return set_last(cudaErrorMemoryAllocation);
    }
    memset(ptr, 0, alloc_size);
    *pHost = ptr;
    pthread_mutex_lock(&g_lock);
    add_alloc_locked(*pHost, size, 1);
    pthread_mutex_unlock(&g_lock);
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size)
{
    return cudaHostAlloc(ptr, size, 0);
}

cudaError_t CUDARTAPI cudaFreeHost(void *ptr)
{
    if (ptr == NULL) {
        return set_last(cudaSuccess);
    }
    pthread_mutex_lock(&g_lock);
    (void)remove_alloc_locked(ptr, NULL);
    pthread_mutex_unlock(&g_lock);
    free(ptr);
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
    (void)flags;
    pthread_mutex_lock(&g_lock);
    add_alloc_locked(ptr, size, 1);
    pthread_mutex_unlock(&g_lock);
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaHostUnregister(void *ptr)
{
    pthread_mutex_lock(&g_lock);
    (void)remove_alloc_locked(ptr, NULL);
    pthread_mutex_unlock(&g_lock);
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)
{
    (void)flags;
    if (pDevice == NULL) {
        return set_last(cudaErrorInvalidValue);
    }
    *pDevice = pHost;
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, int device)
{
    (void)devPtr;
    (void)count;
    (void)advice;
    (void)device;
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    (void)stream;
    cudaError_t err = cudaSuccess;
    if (count == 0 || dst == src) {
        return set_last(cudaSuccess);
    }
    if (kind == cudaMemcpyDefault) {
        pthread_mutex_lock(&g_lock);
        int dst_dev = is_device_ptr_locked(dst);
        int src_dev = is_device_ptr_locked(src);
        pthread_mutex_unlock(&g_lock);
        if (dst_dev && src_dev) {
            kind = cudaMemcpyDeviceToDevice;
        } else if (dst_dev) {
            kind = cudaMemcpyHostToDevice;
        } else if (src_dev) {
            kind = cudaMemcpyDeviceToHost;
        } else {
            kind = cudaMemcpyHostToHost;
        }
    }
    switch (kind) {
    case cudaMemcpyHostToHost:
        memmove(dst, src, count);
        break;
    case cudaMemcpyHostToDevice:
        err = from_cu(cuMemcpyHtoD((CUdeviceptr)(uintptr_t)dst, src, count));
        break;
    case cudaMemcpyDeviceToHost:
        err = from_cu(cuMemcpyDtoH(dst, (CUdeviceptr)(uintptr_t)src, count));
        break;
    case cudaMemcpyDeviceToDevice:
        err = from_cu(cuMemcpyDtoD((CUdeviceptr)(uintptr_t)dst, (CUdeviceptr)(uintptr_t)src, count));
        break;
    default:
        err = cudaErrorInvalidMemcpyDirection;
        break;
    }
    return set_last(err);
}

cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    return cudaMemcpyAsync(dst, src, count, kind, NULL);
}

cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream)
{
    (void)dstDevice;
    (void)srcDevice;
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
}

cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
                                        size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    for (size_t y = 0; y < height; y++) {
        cudaError_t err = cudaMemcpyAsync((char *)dst + y * dpitch, (const char *)src + y * spitch,
                                          width, kind, stream);
        if (err != cudaSuccess) {
            return err;
        }
    }
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream)
{
    (void)p;
    (void)stream;
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream)
{
    (void)stream;
    return set_last(from_cu(cuMemsetD8((CUdeviceptr)(uintptr_t)devPtr, (unsigned char)value, count)));
}

cudaError_t CUDARTAPI cudaMemset(void *devPtr, int value, size_t count)
{
    return cudaMemsetAsync(devPtr, value, count, NULL);
}

cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
    if (pStream == NULL) {
        return set_last(cudaErrorInvalidValue);
    }
    struct lanxin_cudart_stream *s = (struct lanxin_cudart_stream *)calloc(1, sizeof(*s));
    if (s == NULL) {
        return set_last(cudaErrorMemoryAllocation);
    }
    s->magic = LANXIN_CUDART_STREAM_MAGIC;
    s->flags = flags;
    cudaError_t err = from_cu(cuStreamCreate(&s->driver, flags));
    if (err != cudaSuccess) {
        free(s);
        return set_last(err);
    }
    *pStream = (cudaStream_t)s;
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream)
{
    if (stream == NULL || stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
        return set_last(cudaSuccess);
    }
    struct lanxin_cudart_stream *s = (struct lanxin_cudart_stream *)stream;
    if (s->magic == LANXIN_CUDART_STREAM_MAGIC) {
        (void)cuStreamDestroy(s->driver);
        s->magic = 0;
        free(s);
        return set_last(cudaSuccess);
    }
    return set_last(from_cu(cuStreamDestroy((CUstream)stream)));
}

cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream)
{
    return set_last(from_cu(cuStreamSynchronize(driver_stream(stream))));
}

cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    (void)stream;
    (void)event;
    (void)flags;
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus)
{
    (void)stream;
    if (pCaptureStatus == NULL) {
        return set_last(cudaErrorInvalidValue);
    }
    *pCaptureStatus = cudaStreamCaptureStatusNone;
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode)
{
    (void)stream;
    (void)mode;
    return set_last(cudaErrorStreamCaptureUnsupported);
}

cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph)
{
    (void)stream;
    if (pGraph != NULL) {
        *pGraph = NULL;
    }
    return set_last(cudaErrorStreamCaptureUnsupported);
}

cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    if (event == NULL) {
        return set_last(cudaErrorInvalidValue);
    }
    struct lanxin_cudart_event *e = (struct lanxin_cudart_event *)calloc(1, sizeof(*e));
    if (e == NULL) {
        return set_last(cudaErrorMemoryAllocation);
    }
    e->magic = LANXIN_CUDART_EVENT_MAGIC;
    e->flags = flags;
    cudaError_t err = from_cu(cuEventCreate(&e->driver, flags));
    if (err != cudaSuccess) {
        free(e);
        return set_last(err);
    }
    *event = (cudaEvent_t)e;
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event)
{
    if (event == NULL) {
        return set_last(cudaSuccess);
    }
    struct lanxin_cudart_event *e = (struct lanxin_cudart_event *)event;
    if (e->magic == LANXIN_CUDART_EVENT_MAGIC) {
        (void)cuEventDestroy(e->driver);
        e->magic = 0;
        free(e);
        return set_last(cudaSuccess);
    }
    return set_last(from_cu(cuEventDestroy((CUevent)event)));
}

cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    return set_last(from_cu(cuEventRecord(driver_event(event), driver_stream(stream))));
}

cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event)
{
    return set_last(from_cu(cuEventSynchronize(driver_event(event))));
}

cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args,
                                       size_t sharedMem, cudaStream_t stream)
{
    CUfunction cu_func = lookup_function(func);
    if (cu_func == NULL) {
        TRACE("unregistered kernel host=%p grid=%ux%ux%u block=%ux%ux%u\n",
              func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
        return set_last(cudaSuccess);
    }
    CUresult rc = cuLaunchKernel(cu_func, gridDim.x, gridDim.y, gridDim.z,
                                 blockDim.x, blockDim.y, blockDim.z,
                                 (unsigned int)sharedMem, driver_stream(stream), args, NULL);
    return set_last(from_cu(rc));
}

cudaError_t CUDARTAPI cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                                  void **args, size_t sharedMem, cudaStream_t stream)
{
    return cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t CUDARTAPI cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args)
{
    if (config == NULL) {
        return set_last(cudaErrorInvalidValue);
    }
    return cudaLaunchKernel(func, config->gridDim, config->blockDim, args, config->dynamicSmemBytes, config->stream);
}

cudaError_t CUDARTAPI cudaLaunchKernelExC_ptsz(const cudaLaunchConfig_t *config, const void *func, void **args)
{
    return cudaLaunchKernelExC(config, func, args);
}

cudaError_t CUDARTAPI cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value)
{
    (void)func;
    (void)attr;
    (void)value;
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func)
{
    (void)func;
    if (attr == NULL) {
        return set_last(cudaErrorInvalidValue);
    }
    memset(attr, 0, sizeof(*attr));
    attr->maxThreadsPerBlock = 1024;
    attr->numRegs = 64;
    attr->sharedSizeBytes = 0;
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags)
{
    (void)func;
    (void)blockSize;
    (void)dynamicSMemSize;
    (void)flags;
    if (numBlocks == NULL) {
        return set_last(cudaErrorInvalidValue);
    }
    *numBlocks = 1;
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                                           unsigned long long flags)
{
    (void)graph;
    (void)flags;
    if (pGraphExec != NULL) {
        *pGraphExec = (cudaGraphExec_t)calloc(1, 8);
    }
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)
{
    (void)graphExec;
    (void)stream;
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t graph)
{
    free((void *)graph);
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t graphExec)
{
    free((void *)graphExec);
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                                          cudaGraphExecUpdateResultInfo *resultInfo)
{
    (void)hGraphExec;
    (void)hGraph;
    if (resultInfo != NULL) {
        memset(resultInfo, 0, sizeof(*resultInfo));
        resultInfo->result = cudaGraphExecUpdateSuccess;
    }
    return set_last(cudaSuccess);
}

cudaError_t CUDARTAPI cudaGetLastError(void)
{
    cudaError_t err = g_last_error;
    g_last_error = cudaSuccess;
    return err;
}

cudaError_t CUDARTAPI cudaPeekAtLastError(void)
{
    return g_last_error;
}

const char *CUDARTAPI cudaGetErrorString(cudaError_t error)
{
    switch (error) {
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
    case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
    case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle";
    case cudaErrorNotSupported: return "cudaErrorNotSupported";
    case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorStreamCaptureUnsupported: return "cudaErrorStreamCaptureUnsupported";
    default: return "cudaErrorUnknown";
    }
}

unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, struct CUstream_st *stream)
{
    tls_grid = gridDim;
    tls_block = blockDim;
    tls_shared = sharedMem;
    tls_stream = (cudaStream_t)stream;
    tls_has_config = 1;
    return 0;
}

cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream)
{
    if (!tls_has_config) {
        return cudaErrorInvalidValue;
    }
    if (gridDim != NULL) {
        *gridDim = tls_grid;
    }
    if (blockDim != NULL) {
        *blockDim = tls_block;
    }
    if (sharedMem != NULL) {
        *sharedMem = tls_shared;
    }
    if (stream != NULL) {
        *(cudaStream_t *)stream = tls_stream;
    }
    tls_has_config = 0;
    return cudaSuccess;
}

void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin)
{
    struct lanxin_cudart_module *m = (struct lanxin_cudart_module *)calloc(1, sizeof(*m));
    if (m == NULL) {
        return NULL;
    }
    m->handle = (void **)calloc(1, sizeof(void *));
    if (m->handle == NULL) {
        free(m);
        return NULL;
    }
    *m->handle = m;
    (void)ensure_cuda();
    CUresult rc = cuModuleLoadData(&m->module, fatCubin);
    if (rc != CUDA_SUCCESS) {
        m->module = NULL;
    }
    pthread_mutex_lock(&g_lock);
    m->next = g_modules;
    g_modules = m;
    pthread_mutex_unlock(&g_lock);
    TRACE("register fatbin=%p module=%p\n", fatCubin, (void *)m->module);
    return m->handle;
}

void CUDARTAPI __cudaRegisterFatBinaryEnd(void **fatCubinHandle)
{
    (void)fatCubinHandle;
}

void CUDARTAPI __cudaUnregisterFatBinary(void **fatCubinHandle)
{
    if (fatCubinHandle == NULL) {
        return;
    }
    pthread_mutex_lock(&g_lock);
    struct lanxin_cudart_module **prev = &g_modules;
    while (*prev != NULL) {
        struct lanxin_cudart_module *m = *prev;
        if (m->handle == fatCubinHandle) {
            *prev = m->next;
            if (m->module != NULL) {
                (void)cuModuleUnload(m->module);
            }
            free(m->handle);
            free(m);
            break;
        }
        prev = &m->next;
    }
    pthread_mutex_unlock(&g_lock);
}

void CUDARTAPI __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
                                      const char *deviceName, int thread_limit, uint3 *tid,
                                      uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize)
{
    (void)deviceFun;
    (void)thread_limit;
    (void)tid;
    (void)bid;
    (void)bDim;
    (void)gDim;
    (void)wSize;
    CUmodule module = NULL;
    if (fatCubinHandle != NULL) {
        struct lanxin_cudart_module *m = (struct lanxin_cudart_module *)*fatCubinHandle;
        if (m != NULL) {
            module = m->module;
        }
    }
    CUfunction fn = NULL;
    if (module != NULL) {
        const char *name = deviceName != NULL ? deviceName : (const char *)deviceFun;
        if (name != NULL) {
            (void)cuModuleGetFunction(&fn, module, name);
        }
    }
    struct lanxin_cudart_function *f = (struct lanxin_cudart_function *)calloc(1, sizeof(*f));
    if (f == NULL) {
        return;
    }
    f->host = (const void *)hostFun;
    f->function = fn;
    f->name = deviceName != NULL ? strdup(deviceName) : NULL;
    pthread_mutex_lock(&g_lock);
    f->next = g_functions;
    g_functions = f;
    pthread_mutex_unlock(&g_lock);
    TRACE("register function host=%p name=%s fn=%p\n", (const void *)hostFun,
          deviceName != NULL ? deviceName : "<unnamed>", (void *)fn);
}

void CUDARTAPI __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                                 const char *deviceName, int ext, size_t size, int constant, int global)
{
    (void)fatCubinHandle; (void)hostVar; (void)deviceAddress; (void)deviceName;
    (void)ext; (void)size; (void)constant; (void)global;
}

void CUDARTAPI __cudaRegisterManagedVar(void **fatCubinHandle, void **hostVarPtrAddress,
                                        char *deviceAddress, const char *deviceName, int ext,
                                        size_t size, int constant, int global)
{
    (void)fatCubinHandle; (void)hostVarPtrAddress; (void)deviceAddress; (void)deviceName;
    (void)ext; (void)size; (void)constant; (void)global;
}

char CUDARTAPI __cudaInitModule(void **fatCubinHandle)
{
    (void)fatCubinHandle;
    return 0;
}
