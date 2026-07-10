#define _GNU_SOURCE
#include "../include/cuda.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* cuda.h maps many public names to ABI-suffixed entry points. Export both. */
#undef cuDeviceTotalMem
#undef cuCtxCreate
#undef cuCtxDestroy
#undef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxSetFlags
#undef cuModuleGetGlobal
#undef cuLinkCreate
#undef cuLinkAddData
#undef cuLinkAddFile
#undef cuMemGetInfo
#undef cuMemAlloc
#undef cuMemAllocPitch
#undef cuMemFree
#undef cuMemGetAddressRange
#undef cuMemAllocHost
#undef cuMemHostGetDevicePointer
#undef cuMemcpyHtoD
#undef cuMemcpyDtoH
#undef cuMemcpyDtoD
#undef cuMemcpyDtoA
#undef cuMemcpyAtoD
#undef cuMemcpyHtoA
#undef cuMemcpyAtoH
#undef cuMemcpyAtoA
#undef cuMemcpyHtoAAsync
#undef cuMemcpyAtoHAsync
#undef cuMemcpy2D
#undef cuMemcpy2DUnaligned
#undef cuMemcpy3D
#undef cuMemcpyHtoDAsync
#undef cuMemcpyDtoHAsync
#undef cuMemcpyDtoDAsync
#undef cuMemcpy2DAsync
#undef cuMemcpy3DAsync
#undef cuMemsetD8
#undef cuMemsetD16
#undef cuMemsetD32
#undef cuMemsetD2D8
#undef cuMemsetD2D16
#undef cuMemsetD2D32
#undef cuGetProcAddress
#undef cuStreamGetCaptureInfo
#undef cuStreamGetCaptureInfo_v2
#undef cuStreamDestroy
#undef cuLaunchKernel
#undef cuLaunchKernelEx
#undef cuEventRecord
#undef cuEventRecordWithFlags
#undef cuEventDestroy

#define LANXIN_CUDA_VERSION 12090
#define DEFAULT_TOTAL_MEM_MB 32768ULL
#define DEFAULT_SM_COUNT 170
#define DEFAULT_COMPUTE_MAJOR 12
#define DEFAULT_COMPUTE_MINOR 0
#define HANDLE_MAGIC_CTX 0x4c584354u
#define HANDLE_MAGIC_STREAM 0x4c585354u
#define HANDLE_MAGIC_EVENT 0x4c584556u
#define HANDLE_MAGIC_MODULE 0x4c584d4fu
#define HANDLE_MAGIC_FUNC 0x4c58464eu
#define HANDLE_MAGIC_LINK 0x4c584c4eu
#define HANDLE_MAGIC_LIBRARY 0x4c584c49u
#define HANDLE_MAGIC_KERNEL 0x4c584b45u

#define RM_IOCTL_MAGIC 'F'
#define RM_IOCTL_BASE 200
#define RM_ESC_CARD_INFO (RM_IOCTL_BASE + 0)
#define RM_ESC_REGISTER_FD (RM_IOCTL_BASE + 1)
#define RM_ESC_CHECK_VERSION_STR (RM_IOCTL_BASE + 10)
#define RM_ESC_IOCTL_XFER_CMD (RM_IOCTL_BASE + 11)
#define RM_ESC_ATTACH_GPUS_TO_FD (RM_IOCTL_BASE + 12)
#define RM_ESC_WAIT_OPEN_COMPLETE (RM_IOCTL_BASE + 18)
#define RM_ESC_RM_ALLOC_MEMORY 0x27
#define RM_ESC_RM_FREE 0x29
#define RM_ESC_RM_CONTROL 0x2a
#define RM_ESC_RM_ALLOC 0x2b
#define RM_ESC_RM_MAP_MEMORY 0x4e
#define RM_ESC_RM_UNMAP_MEMORY 0x4f
#define RM_ESC_RM_ALLOC_CONTEXT_DMA2 0x54
#define RM_ESC_RM_MAP_MEMORY_DMA 0x57
#define RM_ESC_RM_UNMAP_MEMORY_DMA 0x58
#define RM_NV_OK 0u
#define RM_NV01_NULL_OBJECT 0x0u
#define RM_NV01_ROOT 0x0u
#define RM_NV01_DEVICE_0 0x80u
#define RM_NV20_SUBDEVICE_0 0x2080u
#define RM_NV01_CONTEXT_DMA 0x2u
#define RM_NV01_MEMORY_SYSTEM 0x3eu
#define RM_NV01_MEMORY_VIRTUAL 0x70u
#define RM_BLACKWELL_CHANNEL_GPFIFO_B 0xca6fu
#define RM_BLACKWELL_COMPUTE_B 0xcec0u
#define RM_BLACKWELL_COMPUTE_A 0xcdc0u
#define RM_HOPPER_COMPUTE_A 0xcbc0u
#define RM_AMPERE_COMPUTE_B 0xc7c0u
#define RM_AMPERE_COMPUTE_A 0xc6c0u
#define RM_NV_MAX_SUBDEVICES 8u
#define RM_NV2080_ENGINE_TYPE_GRAPHICS 0x1u
#define RM_NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS (1u << 4)
#define RM_NVOS02_FLAGS_LOCATION_PCI (0u << 8)
#define RM_NVOS02_FLAGS_COHERENCY_WRITE_COMBINE (2u << 12)
#define RM_NVOS02_FLAGS_COHERENCY_WRITE_BACK (5u << 12)
#define RM_NVOS02_FLAGS_GPU_CACHEABLE_YES (1u << 18)
#define RM_NVOS02_FLAGS_MAPPING_NO_MAP (1u << 30)
#define RM_NVOS03_FLAGS_MAPPING_KERNEL (1u << 20)
#define RM_NVOS03_FLAGS_HASH_TABLE_DISABLE (1u << 29)
#define RM_NVOS04_FLAGS_MAP_CHANNEL_TRUE (1u << 30)
#define RM_NVOS46_FLAGS_CACHE_SNOOP_ENABLE (1u << 4)
#define RM_NVOS46_FLAGS_PAGE_SIZE_4KB (1u << 8)
#define RM_NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES 0u
#define RM_NVA06F_CTRL_CMD_GPFIFO_SCHEDULE 0xa06f0103u
#define RM_NVA06F_CTRL_CMD_BIND 0xa06f0104u
#define RM_NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN 0xc36f0108u
#define RM_NV906F_CTRL_GET_CLASS_ENGINEID 0x906f0101u
#define RM_NVA06F_SUBCHANNEL_COMPUTE 1u
#define RM_NVC46F_DMA_SEC_OP_INC_METHOD 1u
#define RM_COMPUTE_SET_OBJECT 0x0000u
#define RM_COMPUTE_NO_OPERATION 0x0100u
#define RM_COMPUTE_PIPE_NOP 0x1a2cu

typedef uint8_t rm_bool;
typedef uint32_t rm_u32;
typedef int32_t rm_v32;
typedef uint64_t rm_u64;
typedef uint64_t rm_p64;
typedef uint32_t rm_handle;

typedef struct {
    rm_u32 domain;
    uint8_t bus;
    uint8_t slot;
    uint8_t function;
    uint16_t vendor_id;
    uint16_t device_id;
} rm_pci_info_t;

typedef struct {
    rm_u32 cmd;
    rm_u32 size;
    rm_p64 ptr __attribute__((aligned(8)));
} rm_ioctl_xfer_t;

typedef struct {
    uint8_t valid;
    rm_pci_info_t pci_info;
    rm_u32 gpu_id;
    uint16_t interrupt_line;
    rm_u64 reg_address __attribute__((aligned(8)));
    rm_u64 reg_size __attribute__((aligned(8)));
    rm_u64 fb_address __attribute__((aligned(8)));
    rm_u64 fb_size __attribute__((aligned(8)));
    rm_u32 minor_number;
    uint8_t dev_name[10];
} rm_ioctl_card_info_t;

typedef struct {
    rm_u32 cmd;
    rm_u32 reply;
    char versionString[64];
} rm_ioctl_version_t;

typedef struct {
    int rc;
    rm_u32 adapterStatus;
} rm_ioctl_wait_open_complete_t;

typedef struct {
    int ctl_fd;
} rm_ioctl_register_fd_t;

typedef struct {
    rm_handle hRoot;
    rm_handle hObjectParent;
    rm_handle hObjectOld;
    rm_v32 status;
} rm_nvos00_t;

typedef struct {
    rm_handle hRoot;
    rm_handle hObjectParent;
    rm_handle hObjectNew;
    rm_v32 hClass;
    rm_v32 flags;
    rm_p64 pMemory __attribute__((aligned(8)));
    rm_u64 limit __attribute__((aligned(8)));
    rm_v32 status;
} rm_nvos02_t;

typedef struct {
    rm_nvos02_t params;
    int fd;
} rm_nvos02_with_fd_t;

typedef struct {
    rm_handle hRoot;
    rm_handle hObjectParent;
    rm_handle hObjectNew;
    rm_v32 hClass;
    rm_p64 pAllocParms __attribute__((aligned(8)));
    rm_u32 paramsSize;
    rm_v32 status;
} rm_nvos21_t;

typedef struct {
    rm_handle hClient;
    rm_handle hDevice;
    rm_handle hMemory;
    rm_u64 offset __attribute__((aligned(8)));
    rm_u64 length __attribute__((aligned(8)));
    rm_p64 pLinearAddress __attribute__((aligned(8)));
    rm_u32 status;
    rm_u32 flags;
} rm_nvos33_t;

typedef struct {
    rm_nvos33_t params;
    int fd;
} rm_nvos33_with_fd_t;

typedef struct {
    rm_handle hClient;
    rm_handle hDevice;
    rm_handle hMemory;
    rm_p64 pLinearAddress __attribute__((aligned(8)));
    rm_u32 status;
    rm_u32 flags;
} rm_nvos34_t;

typedef struct {
    rm_handle hObjectParent;
    rm_handle hSubDevice;
    rm_handle hObjectNew;
    rm_v32 hClass;
    rm_v32 flags;
    rm_u32 selector;
    rm_handle hMemory;
    rm_u64 offset __attribute__((aligned(8)));
    rm_u64 limit __attribute__((aligned(8)));
    rm_v32 status;
} rm_nvos39_t;

typedef struct {
    rm_handle hClient;
    rm_handle hDevice;
    rm_handle hDma;
    rm_handle hMemory;
    rm_u64 offset __attribute__((aligned(8)));
    rm_u64 length __attribute__((aligned(8)));
    rm_v32 flags;
    rm_u64 dmaOffset __attribute__((aligned(8)));
    rm_v32 status;
} rm_nvos46_t;

typedef struct {
    rm_handle hClient;
    rm_handle hDevice;
    rm_handle hDma;
    rm_handle hMemory;
    rm_v32 flags;
    rm_u64 dmaOffset __attribute__((aligned(8)));
    rm_u64 size __attribute__((aligned(8)));
    rm_v32 status;
} rm_nvos47_t;

typedef struct {
    rm_handle hClient;
    rm_handle hObject;
    rm_v32 cmd;
    rm_u32 flags;
    rm_p64 params __attribute__((aligned(8)));
    rm_u32 paramsSize;
    rm_v32 status;
} rm_nvos54_t;

typedef struct {
    rm_u32 deviceId;
    rm_handle hClientShare;
    rm_handle hTargetClient;
    rm_handle hTargetDevice;
    rm_v32 flags;
    rm_u64 vaSpaceSize __attribute__((aligned(8)));
    rm_u64 vaStartInternal __attribute__((aligned(8)));
    rm_u64 vaLimitInternal __attribute__((aligned(8)));
    rm_v32 vaMode;
} rm_nv0080_alloc_params_t;

typedef struct {
    rm_u32 subDeviceId;
} rm_nv2080_alloc_params_t;

typedef struct {
    rm_u64 offset __attribute__((aligned(8)));
    rm_u64 limit __attribute__((aligned(8)));
    rm_handle hVASpace;
} rm_memory_virtual_params_t;

typedef struct {
    rm_u64 base __attribute__((aligned(8)));
    rm_u64 size __attribute__((aligned(8)));
    rm_u32 addressSpace;
    rm_u32 cacheAttrib;
} rm_memory_desc_params_t;

typedef struct {
    rm_handle hObjectError;
    rm_handle hObjectBuffer;
    rm_u64 gpFifoOffset __attribute__((aligned(8)));
    rm_u32 gpFifoEntries;
    rm_u32 flags;
    rm_handle hContextShare;
    rm_handle hVASpace;
    rm_handle hUserdMemory[RM_NV_MAX_SUBDEVICES];
    rm_u64 userdOffset[RM_NV_MAX_SUBDEVICES] __attribute__((aligned(8)));
    rm_u32 engineType;
    rm_u32 cid;
    rm_u32 subDeviceId;
    rm_handle hObjectEccError;
    rm_memory_desc_params_t instanceMem __attribute__((aligned(8)));
    rm_memory_desc_params_t userdMem __attribute__((aligned(8)));
    rm_memory_desc_params_t ramfcMem __attribute__((aligned(8)));
    rm_memory_desc_params_t mthdbufMem __attribute__((aligned(8)));
    rm_handle hPhysChannelGroup;
    rm_u32 internalFlags;
    rm_memory_desc_params_t errorNotifierMem __attribute__((aligned(8)));
    rm_memory_desc_params_t eccErrorNotifierMem __attribute__((aligned(8)));
    rm_u32 ProcessID;
    rm_u32 SubProcessID;
    rm_u32 encryptIv[3];
    rm_u32 decryptIv[3];
    rm_u32 hmacNonce[8];
} rm_channel_alloc_params_t;

typedef struct {
    rm_u32 engineType;
} rm_channel_bind_params_t;

typedef struct {
    rm_bool bEnable;
    rm_bool bSkipSubmit;
    rm_bool bSkipEnable;
} rm_channel_schedule_params_t;

typedef struct {
    rm_u32 workSubmitToken;
} rm_channel_token_params_t;

typedef struct {
    rm_handle hObject;
    rm_u32 classEngineID;
    rm_u32 classID;
    rm_u32 engineID;
} rm_get_class_engine_id_params_t;

struct CUctx_st {
    uint32_t magic;
    CUdevice dev;
    unsigned int flags;
    unsigned long long id;
    bool primary;
};

struct CUstream_st {
    uint32_t magic;
    unsigned int flags;
    int priority;
    unsigned long long id;
    CUcontext ctx;
};

struct CUevent_st {
    uint32_t magic;
    unsigned int flags;
    bool recorded;
    struct timespec stamp;
};

struct CUmod_st {
    uint32_t magic;
    char *name;
    const void *image;
    void *owned_image;
    size_t image_size;
    struct CUfunc_st *functions;
    struct module_global *globals;
    struct CUmod_st *next;
};

struct CUfunc_st {
    uint32_t magic;
    char *name;
    CUmodule module;
    struct CUfunc_st *next;
};

struct CUlinkState_st {
    uint32_t magic;
    unsigned int input_count;
    void *image;
    size_t image_size;
};

struct CUlib_st {
    uint32_t magic;
    CUmodule module;
    struct CUkern_st *kernels;
};

struct CUkern_st {
    uint32_t magic;
    char *name;
    CUlibrary library;
    CUfunction function;
    struct CUkern_st *next;
};

struct module_global {
    char *name;
    CUdeviceptr dptr;
    size_t bytes;
    struct module_global *next;
};

struct allocation {
    CUdeviceptr dptr;
    void *host;
    size_t size;
    size_t mapped_size;
    unsigned int memory_type;
    bool owns_host;
    bool rm_backed;
    rm_handle rm_memory;
    rm_p64 rm_linear;
    unsigned long long id;
    struct allocation *next;
};

struct driver_state {
    bool initialized;
    CUresult init_result;
    int ctl_fd;
    int gpu_fd;
    int uvm_fd;
    int uvm_tools_fd;
    int rm_gpfifo_fd;
    int rm_userd_fd;
    int device_count;
    char name[128];
    char bus_id[32];
    char uuid_text[64];
    CUuuid uuid;
    size_t total_mem;
    size_t allocated_bytes;
    int sm_count;
    int cc_major;
    int cc_minor;
    unsigned int primary_flags;
    CUcontext primary_ctx;
    unsigned long long next_id;
    rm_handle next_rm_handle;
    bool rm_attempted;
    bool rm_ready;
    rm_u32 rm_gpu_id;
    rm_handle rm_client;
    rm_handle rm_device;
    rm_handle rm_subdevice;
    bool rm_channel_attempted;
    bool rm_channel_ready;
    rm_handle rm_vaspace;
    rm_handle rm_notifier;
    rm_handle rm_error_ctxdma;
    rm_handle rm_gpfifo;
    rm_handle rm_userd;
    rm_handle rm_channel;
    rm_handle rm_compute;
    rm_p64 rm_notifier_va;
    rm_p64 rm_gpfifo_va;
    rm_p64 rm_gpfifo_linear;
    rm_p64 rm_userd_linear;
    void *rm_gpfifo_cpu;
    void *rm_userd_cpu;
    rm_u32 rm_work_submit_token;
    rm_u32 rm_gpfifo_put;
    rm_u32 rm_compute_class;
    rm_u32 rm_compute_class_engine_id;
    struct allocation *allocs;
    CUmodule modules;
};

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static struct driver_state g = {
    .ctl_fd = -1,
    .gpu_fd = -1,
    .uvm_fd = -1,
    .uvm_tools_fd = -1,
    .rm_gpfifo_fd = -1,
    .rm_userd_fd = -1,
    .device_count = 0,
    .name = "NVIDIA GPU",
    .bus_id = "0001:01:00.0",
    .uuid_text = "",
    .total_mem = DEFAULT_TOTAL_MEM_MB * 1024ULL * 1024ULL,
    .sm_count = DEFAULT_SM_COUNT,
    .cc_major = DEFAULT_COMPUTE_MAJOR,
    .cc_minor = DEFAULT_COMPUTE_MINOR,
    .next_id = 1,
    .next_rm_handle = 0x3000,
};

static __thread CUcontext tls_current_ctx;
static __thread CUcontext tls_ctx_stack[32];
static __thread int tls_ctx_depth;

static int env_enabled(const char *name)
{
    const char *value = getenv(name);
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static unsigned long long env_ull(const char *name, unsigned long long fallback)
{
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 0);
    if (errno != 0 || end == value) {
        return fallback;
    }
    return parsed;
}

static void tracef(const char *fmt, ...)
{
    if (!env_enabled("LANXIN_NVIDIA_CUDA_TRACE")) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    fputs("[lanxin-libcuda] ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

static int env_disabled(const char *name)
{
    const char *value = getenv(name);
    return value != NULL && strcmp(value, "0") == 0;
}

static unsigned long rm_iowr(unsigned int nr, size_t size)
{
    return _IOC(_IOC_READ | _IOC_WRITE, RM_IOCTL_MAGIC, nr, size);
}

static int rm_ioctl_direct(int fd, rm_u32 cmd, void *data, size_t size)
{
    return ioctl(fd, rm_iowr(cmd, size), data);
}

static int rm_ioctl_xfer(int fd, rm_u32 cmd, void *data, size_t size)
{
    rm_ioctl_xfer_t xfer = {
        .cmd = cmd,
        .size = (rm_u32)size,
        .ptr = (rm_p64)(uintptr_t)data,
    };
    return ioctl(fd, rm_iowr(RM_ESC_IOCTL_XFER_CMD, sizeof(xfer)), &xfer);
}

static int rm_alloc_object_locked(rm_handle hRoot, rm_handle hParent, rm_handle *hObject,
                                  rm_u32 hClass, void *params, rm_u32 paramsSize,
                                  const char *label)
{
    rm_nvos21_t api;
    memset(&api, 0, sizeof(api));
    api.hRoot = hRoot;
    api.hObjectParent = hParent;
    api.hObjectNew = hObject != NULL ? *hObject : 0;
    api.hClass = (rm_v32)hClass;
    api.pAllocParms = (rm_p64)(uintptr_t)params;
    api.paramsSize = paramsSize;
    int rc = rm_ioctl_xfer(g.ctl_fd, RM_ESC_RM_ALLOC, &api, sizeof(api));
    tracef("%s ioctl=%d errno=%d status=0x%08x object=0x%x class=0x%x",
           label, rc, rc == 0 ? 0 : errno, (rm_u32)api.status, api.hObjectNew, hClass);
    if (rc == 0 && (rm_u32)api.status == RM_NV_OK) {
        if (hObject != NULL) {
            *hObject = api.hObjectNew;
        }
        return 0;
    }
    return -1;
}

static int rm_free_object_locked(rm_handle hRoot, rm_handle hParent, rm_handle hObject)
{
    if (hObject == 0) {
        return 0;
    }
    rm_nvos00_t api;
    memset(&api, 0, sizeof(api));
    api.hRoot = hRoot;
    api.hObjectParent = hParent;
    api.hObjectOld = hObject;
    int rc = rm_ioctl_xfer(g.ctl_fd, RM_ESC_RM_FREE, &api, sizeof(api));
    tracef("rm_free object=0x%x ioctl=%d errno=%d status=0x%08x",
           hObject, rc, rc == 0 ? 0 : errno, (rm_u32)api.status);
    return rc == 0 && (rm_u32)api.status == RM_NV_OK ? 0 : -1;
}

static int rm_alloc_memory_locked(int ioctl_fd, int map_fd, rm_handle *hMemory,
                                  rm_u32 hClass, rm_u32 flags, size_t size,
                                  const char *label)
{
    rm_nvos02_with_fd_t api;
    memset(&api, 0, sizeof(api));
    api.params.hRoot = g.rm_client;
    api.params.hObjectParent = g.rm_device;
    api.params.hObjectNew = *hMemory;
    api.params.hClass = (rm_v32)hClass;
    api.params.flags = (rm_v32)flags;
    api.params.limit = (rm_u64)size - 1U;
    api.fd = map_fd;
    int rc = rm_ioctl_xfer(ioctl_fd, RM_ESC_RM_ALLOC_MEMORY, &api, sizeof(api));
    tracef("%s ioctl=%d errno=%d status=0x%08x handle=0x%x size=%zu",
           label, rc, rc == 0 ? 0 : errno, (rm_u32)api.params.status,
           api.params.hObjectNew, size);
    if (rc == 0 && (rm_u32)api.params.status == RM_NV_OK) {
        *hMemory = api.params.hObjectNew;
        return 0;
    }
    return -1;
}

static int rm_map_cpu_locked(int map_fd, rm_handle hMemory, size_t size,
                             rm_p64 *linear, void **cpu, const char *label)
{
    rm_nvos33_with_fd_t api;
    memset(&api, 0, sizeof(api));
    api.params.hClient = g.rm_client;
    api.params.hDevice = g.rm_device;
    api.params.hMemory = hMemory;
    api.params.length = size;
    api.fd = map_fd;
    int rc = rm_ioctl_xfer(g.ctl_fd, RM_ESC_RM_MAP_MEMORY, &api, sizeof(api));
    tracef("%s map ioctl=%d errno=%d status=0x%08x linear=0x%llx",
           label, rc, rc == 0 ? 0 : errno, api.params.status,
           (unsigned long long)api.params.pLinearAddress);
    if (rc != 0 || api.params.status != RM_NV_OK) {
        return -1;
    }
    void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, map_fd, 0);
    if (addr == MAP_FAILED) {
        tracef("%s mmap failed errno=%d", label, errno);
        rm_nvos34_t unmap_api;
        memset(&unmap_api, 0, sizeof(unmap_api));
        unmap_api.hClient = g.rm_client;
        unmap_api.hDevice = g.rm_device;
        unmap_api.hMemory = hMemory;
        unmap_api.pLinearAddress = api.params.pLinearAddress;
        (void)rm_ioctl_xfer(g.ctl_fd, RM_ESC_RM_UNMAP_MEMORY, &unmap_api, sizeof(unmap_api));
        return -1;
    }
    *linear = api.params.pLinearAddress;
    *cpu = addr;
    return 0;
}

static void rm_unmap_cpu_locked(rm_handle hMemory, rm_p64 linear, void *cpu, size_t size)
{
    if (cpu != NULL && size != 0) {
        munmap(cpu, size);
    }
    if (hMemory != 0 && linear != 0) {
        rm_nvos34_t api;
        memset(&api, 0, sizeof(api));
        api.hClient = g.rm_client;
        api.hDevice = g.rm_device;
        api.hMemory = hMemory;
        api.pLinearAddress = linear;
        int rc = rm_ioctl_xfer(g.ctl_fd, RM_ESC_RM_UNMAP_MEMORY, &api, sizeof(api));
        tracef("RM unmap cpu memory=0x%x ioctl=%d errno=%d status=0x%08x",
               hMemory, rc, rc == 0 ? 0 : errno, api.status);
    }
}

static int rm_map_dma_locked(rm_handle hDma, rm_handle hMemory, size_t size,
                             rm_u64 *gpu_va, const char *label)
{
    rm_nvos46_t api;
    memset(&api, 0, sizeof(api));
    api.hClient = g.rm_client;
    api.hDevice = g.rm_device;
    api.hDma = hDma;
    api.hMemory = hMemory;
    api.length = size;
    api.flags = (rm_v32)(RM_NVOS46_FLAGS_PAGE_SIZE_4KB | RM_NVOS46_FLAGS_CACHE_SNOOP_ENABLE);
    api.dmaOffset = *gpu_va;
    int rc = rm_ioctl_xfer(g.ctl_fd, RM_ESC_RM_MAP_MEMORY_DMA, &api, sizeof(api));
    tracef("%s dma map ioctl=%d errno=%d status=0x%08x va=0x%llx",
           label, rc, rc == 0 ? 0 : errno, (rm_u32)api.status,
           (unsigned long long)api.dmaOffset);
    if (rc == 0 && (rm_u32)api.status == RM_NV_OK) {
        *gpu_va = api.dmaOffset;
        return 0;
    }
    return -1;
}

static void rm_unmap_dma_locked(rm_handle hDma, rm_handle hMemory, rm_u64 gpu_va, size_t size)
{
    if (hDma == 0 || hMemory == 0 || gpu_va == 0) {
        return;
    }
    rm_nvos47_t api;
    memset(&api, 0, sizeof(api));
    api.hClient = g.rm_client;
    api.hDevice = g.rm_device;
    api.hDma = hDma;
    api.hMemory = hMemory;
    api.dmaOffset = gpu_va;
    api.size = size;
    int rc = rm_ioctl_xfer(g.ctl_fd, RM_ESC_RM_UNMAP_MEMORY_DMA, &api, sizeof(api));
    tracef("RM unmap dma memory=0x%x va=0x%llx ioctl=%d errno=%d status=0x%08x",
           hMemory, (unsigned long long)gpu_va, rc, rc == 0 ? 0 : errno, (rm_u32)api.status);
}

static int rm_alloc_ctxdma_locked(rm_handle hMemory, rm_u64 limit, rm_handle *ctxdma)
{
    rm_nvos39_t api;
    memset(&api, 0, sizeof(api));
    api.hObjectParent = g.rm_client;
    api.hSubDevice = g.rm_subdevice;
    api.hObjectNew = *ctxdma;
    api.hClass = RM_NV01_CONTEXT_DMA;
    api.flags = (rm_v32)(RM_NVOS03_FLAGS_MAPPING_KERNEL | RM_NVOS03_FLAGS_HASH_TABLE_DISABLE);
    api.hMemory = hMemory;
    api.limit = limit;
    int rc = rm_ioctl_xfer(g.ctl_fd, RM_ESC_RM_ALLOC_CONTEXT_DMA2, &api, sizeof(api));
    tracef("RM alloc ctxdma ioctl=%d errno=%d status=0x%08x ctxdma=0x%x memory=0x%x",
           rc, rc == 0 ? 0 : errno, (rm_u32)api.status, api.hObjectNew, hMemory);
    if (rc == 0 && (rm_u32)api.status == RM_NV_OK) {
        *ctxdma = api.hObjectNew;
        return 0;
    }
    return -1;
}

static int rm_control_locked(rm_handle hObject, rm_u32 cmd, void *params,
                             rm_u32 paramsSize, const char *label)
{
    rm_nvos54_t api;
    memset(&api, 0, sizeof(api));
    api.hClient = g.rm_client;
    api.hObject = hObject;
    api.cmd = (rm_v32)cmd;
    api.params = (rm_p64)(uintptr_t)params;
    api.paramsSize = paramsSize;
    int rc = rm_ioctl_xfer(g.ctl_fd, RM_ESC_RM_CONTROL, &api, sizeof(api));
    tracef("%s control cmd=0x%08x ioctl=%d errno=%d status=0x%08x",
           label, cmd, rc, rc == 0 ? 0 : errno, (rm_u32)api.status);
    return rc == 0 && (rm_u32)api.status == RM_NV_OK ? 0 : -1;
}

static int rm_init_locked(void)
{
    if (g.rm_attempted) {
        return g.rm_ready ? 0 : -1;
    }
    g.rm_attempted = true;

    if (env_disabled("LANXIN_NVIDIA_CUDA_RM")) {
        tracef("RM backend disabled by LANXIN_NVIDIA_CUDA_RM=0");
        return -1;
    }
    if (g.ctl_fd < 0 || g.gpu_fd < 0) {
        return -1;
    }

    rm_ioctl_wait_open_complete_t wait_params = {0};
    int rc = rm_ioctl_direct(g.gpu_fd, RM_ESC_WAIT_OPEN_COMPLETE, &wait_params, sizeof(wait_params));
    if (rc != 0 || wait_params.rc != 0 || wait_params.adapterStatus != RM_NV_OK) {
        tracef("RM wait_open failed ioctl=%d errno=%d open_rc=%d adapter=0x%08x",
               rc, rc == 0 ? 0 : errno, wait_params.rc, wait_params.adapterStatus);
        return -1;
    }

    rm_ioctl_register_fd_t reg_fd = {.ctl_fd = g.ctl_fd};
    rc = rm_ioctl_direct(g.gpu_fd, RM_ESC_REGISTER_FD, &reg_fd, sizeof(reg_fd));
    if (rc != 0) {
        tracef("RM register_fd failed errno=%d", errno);
        return -1;
    }

    rm_ioctl_version_t version = {.cmd = '2'};
    if (rm_ioctl_direct(g.ctl_fd, RM_ESC_CHECK_VERSION_STR, &version, sizeof(version)) == 0) {
        tracef("RM version reply=%u string=%s", version.reply, version.versionString);
    }

    rm_ioctl_card_info_t cards[32];
    memset(cards, 0, sizeof(cards));
    rc = rm_ioctl_direct(g.ctl_fd, RM_ESC_CARD_INFO, cards, sizeof(cards));
    if (rc != 0) {
        tracef("RM card_info failed errno=%d", errno);
        return -1;
    }
    for (size_t i = 0; i < sizeof(cards) / sizeof(cards[0]); i++) {
        if (cards[i].valid) {
            g.rm_gpu_id = cards[i].gpu_id;
            break;
        }
    }
    if (g.rm_gpu_id != 0) {
        rm_u32 attach[1] = {g.rm_gpu_id};
        rc = rm_ioctl_direct(g.ctl_fd, RM_ESC_ATTACH_GPUS_TO_FD, attach, sizeof(attach));
        if (rc != 0) {
            tracef("RM attach_gpus failed errno=%d gpu_id=0x%08x", errno, g.rm_gpu_id);
            return -1;
        }
    }

    rm_handle client = 0;
    rm_u32 root_out = 0;
    if (rm_alloc_object_locked(RM_NV01_NULL_OBJECT, RM_NV01_NULL_OBJECT, &client,
                               RM_NV01_ROOT, &root_out, sizeof(root_out), "rm_alloc_root") != 0) {
        return -1;
    }
    if (client == 0 && root_out != 0) {
        client = root_out;
    }
    if (client == 0) {
        return -1;
    }

    rm_handle device = 0x1000;
    rm_nv0080_alloc_params_t dev_params;
    memset(&dev_params, 0, sizeof(dev_params));
    dev_params.deviceId = 0;
    dev_params.hClientShare = client;
    dev_params.vaMode = RM_NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES;
    if (rm_alloc_object_locked(client, client, &device, RM_NV01_DEVICE_0,
                               &dev_params, sizeof(dev_params), "rm_alloc_device") != 0) {
        rm_free_object_locked(client, client, client);
        return -1;
    }

    rm_handle subdevice = 0x2000;
    rm_nv2080_alloc_params_t sub_params = {.subDeviceId = 0};
    if (rm_alloc_object_locked(client, device, &subdevice, RM_NV20_SUBDEVICE_0,
                               &sub_params, sizeof(sub_params), "rm_alloc_subdevice") != 0) {
        tracef("RM subdevice alloc failed; continuing with device object only");
        subdevice = 0;
    }

    g.rm_client = client;
    g.rm_device = device;
    g.rm_subdevice = subdevice;
    g.rm_ready = true;
    tracef("RM ready client=0x%x device=0x%x subdevice=0x%x gpu_id=0x%08x",
           g.rm_client, g.rm_device, g.rm_subdevice, g.rm_gpu_id);
    return 0;
}

static size_t page_align_size(size_t size)
{
    long page = sysconf(_SC_PAGESIZE);
    size_t align = page > 0 ? (size_t)page : 4096U;
    size_t actual = size == 0 ? 1 : size;
    return (actual + align - 1U) & ~(align - 1U);
}

static void rm_channel_destroy_locked(void)
{
    size_t page = page_align_size(4096);
    if (g.rm_compute != 0) {
        rm_free_object_locked(g.rm_client, g.rm_channel, g.rm_compute);
    }
    if (g.rm_channel != 0) {
        rm_free_object_locked(g.rm_client, g.rm_device, g.rm_channel);
    }
    rm_unmap_cpu_locked(g.rm_userd, g.rm_userd_linear, g.rm_userd_cpu, page);
    if (g.rm_userd != 0) {
        rm_free_object_locked(g.rm_client, g.rm_device, g.rm_userd);
    }
    rm_unmap_dma_locked(g.rm_vaspace, g.rm_gpfifo, g.rm_gpfifo_va, page);
    rm_unmap_cpu_locked(g.rm_gpfifo, g.rm_gpfifo_linear, g.rm_gpfifo_cpu, page);
    if (g.rm_gpfifo != 0) {
        rm_free_object_locked(g.rm_client, g.rm_device, g.rm_gpfifo);
    }
    if (g.rm_error_ctxdma != 0) {
        rm_free_object_locked(g.rm_client, g.rm_device, g.rm_error_ctxdma);
    }
    rm_unmap_dma_locked(g.rm_vaspace, g.rm_notifier, g.rm_notifier_va, page);
    if (g.rm_notifier != 0) {
        rm_free_object_locked(g.rm_client, g.rm_device, g.rm_notifier);
    }
    if (g.rm_vaspace != 0) {
        rm_free_object_locked(g.rm_client, g.rm_device, g.rm_vaspace);
    }
    if (g.rm_gpfifo_fd >= 0) {
        close(g.rm_gpfifo_fd);
    }
    if (g.rm_userd_fd >= 0) {
        close(g.rm_userd_fd);
    }
    g.rm_gpfifo_fd = -1;
    g.rm_userd_fd = -1;
    g.rm_vaspace = 0;
    g.rm_notifier = 0;
    g.rm_error_ctxdma = 0;
    g.rm_gpfifo = 0;
    g.rm_userd = 0;
    g.rm_channel = 0;
    g.rm_compute = 0;
    g.rm_notifier_va = 0;
    g.rm_gpfifo_va = 0;
    g.rm_gpfifo_linear = 0;
    g.rm_userd_linear = 0;
    g.rm_gpfifo_cpu = NULL;
    g.rm_userd_cpu = NULL;
    g.rm_work_submit_token = 0;
    g.rm_gpfifo_put = 0;
    g.rm_compute_class = 0;
    g.rm_compute_class_engine_id = 0;
    g.rm_channel_ready = false;
}

static int rm_channel_init_locked(void)
{
    if (g.rm_channel_ready) {
        return 0;
    }
    if (g.rm_channel_attempted) {
        return -1;
    }
    g.rm_channel_attempted = true;

    if (rm_init_locked() != 0 || !g.rm_ready) {
        return -1;
    }

    size_t page = page_align_size(4096);
    rm_u32 sys_flags = RM_NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS |
                       RM_NVOS02_FLAGS_LOCATION_PCI |
                       RM_NVOS02_FLAGS_COHERENCY_WRITE_COMBINE |
                       RM_NVOS02_FLAGS_GPU_CACHEABLE_YES |
                       RM_NVOS02_FLAGS_MAPPING_NO_MAP;
    rm_u32 coherent_flags = RM_NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS |
                            RM_NVOS02_FLAGS_LOCATION_PCI |
                            RM_NVOS02_FLAGS_COHERENCY_WRITE_BACK |
                            RM_NVOS02_FLAGS_GPU_CACHEABLE_YES |
                            RM_NVOS02_FLAGS_MAPPING_NO_MAP;

    g.rm_gpfifo_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    g.rm_userd_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    if (g.rm_gpfifo_fd < 0 || g.rm_userd_fd < 0) {
        tracef("RM channel fd open failed gpfifo=%d userd=%d errno=%d",
               g.rm_gpfifo_fd, g.rm_userd_fd, errno);
        goto fail;
    }

    g.rm_vaspace = g.next_rm_handle++;
    rm_memory_virtual_params_t va_params;
    memset(&va_params, 0, sizeof(va_params));
    if (rm_alloc_object_locked(g.rm_client, g.rm_device, &g.rm_vaspace,
                               RM_NV01_MEMORY_VIRTUAL, &va_params, sizeof(va_params),
                               "rm_alloc_vaspace") != 0) {
        goto fail;
    }

    g.rm_notifier = g.next_rm_handle++;
    if (rm_alloc_memory_locked(g.gpu_fd, g.ctl_fd, &g.rm_notifier,
                               RM_NV01_MEMORY_SYSTEM, coherent_flags, page,
                               "rm_alloc_notifier") != 0 ||
        rm_map_dma_locked(g.rm_vaspace, g.rm_notifier, page, &g.rm_notifier_va,
                          "rm_map_notifier_dma") != 0) {
        goto fail;
    }

    g.rm_error_ctxdma = g.next_rm_handle++;
    if (rm_alloc_ctxdma_locked(g.rm_notifier, 0xff, &g.rm_error_ctxdma) != 0) {
        goto fail;
    }

    g.rm_gpfifo = g.next_rm_handle++;
    if (rm_alloc_memory_locked(g.gpu_fd, g.rm_gpfifo_fd, &g.rm_gpfifo,
                               RM_NV01_MEMORY_SYSTEM, sys_flags, page,
                               "rm_alloc_gpfifo") != 0 ||
        rm_map_cpu_locked(g.rm_gpfifo_fd, g.rm_gpfifo, page, &g.rm_gpfifo_linear,
                          &g.rm_gpfifo_cpu, "rm_map_gpfifo_cpu") != 0 ||
        rm_map_dma_locked(g.rm_vaspace, g.rm_gpfifo, page, &g.rm_gpfifo_va,
                          "rm_map_gpfifo_dma") != 0) {
        goto fail;
    }
    memset(g.rm_gpfifo_cpu, 0, page);

    g.rm_userd = g.next_rm_handle++;
    if (rm_alloc_memory_locked(g.gpu_fd, g.rm_userd_fd, &g.rm_userd,
                               RM_NV01_MEMORY_SYSTEM, sys_flags, page,
                               "rm_alloc_userd") != 0 ||
        rm_map_cpu_locked(g.rm_userd_fd, g.rm_userd, page, &g.rm_userd_linear,
                          &g.rm_userd_cpu, "rm_map_userd_cpu") != 0) {
        goto fail;
    }
    memset(g.rm_userd_cpu, 0, page);

    rm_channel_alloc_params_t ch_params;
    memset(&ch_params, 0, sizeof(ch_params));
    ch_params.hObjectError = g.rm_error_ctxdma;
    ch_params.hObjectBuffer = g.rm_vaspace;
    ch_params.gpFifoOffset = g.rm_gpfifo_va;
    ch_params.gpFifoEntries = 32;
    ch_params.flags = RM_NVOS04_FLAGS_MAP_CHANNEL_TRUE;
    ch_params.hUserdMemory[0] = g.rm_userd;
    ch_params.engineType = RM_NV2080_ENGINE_TYPE_GRAPHICS;
    ch_params.ProcessID = (rm_u32)getpid();
    g.rm_channel = g.next_rm_handle++;
    if (rm_alloc_object_locked(g.rm_client, g.rm_device, &g.rm_channel,
                               RM_BLACKWELL_CHANNEL_GPFIFO_B, &ch_params, sizeof(ch_params),
                               "rm_alloc_channel_ca6f") != 0) {
        goto fail;
    }

    rm_channel_bind_params_t bind = {.engineType = RM_NV2080_ENGINE_TYPE_GRAPHICS};
    rm_channel_schedule_params_t schedule = {.bEnable = 1, .bSkipSubmit = 0, .bSkipEnable = 0};
    rm_channel_token_params_t token;
    memset(&token, 0, sizeof(token));
    if (rm_control_locked(g.rm_channel, RM_NVA06F_CTRL_CMD_BIND,
                          &bind, sizeof(bind), "rm_channel_bind") != 0 ||
        rm_control_locked(g.rm_channel, RM_NVA06F_CTRL_CMD_GPFIFO_SCHEDULE,
                          &schedule, sizeof(schedule), "rm_channel_schedule") != 0 ||
        rm_control_locked(g.rm_channel, RM_NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
                          &token, sizeof(token), "rm_channel_token") != 0) {
        goto fail;
    }
    g.rm_work_submit_token = token.workSubmitToken;
    g.rm_channel_ready = true;
    tracef("RM channel ready channel=0x%x vaspace=0x%x gpfifo=0x%x/0x%llx userd=0x%x token=0x%08x",
           g.rm_channel, g.rm_vaspace, g.rm_gpfifo,
           (unsigned long long)g.rm_gpfifo_va, g.rm_userd, g.rm_work_submit_token);
    return 0;

fail:
    tracef("RM channel init failed");
    rm_channel_destroy_locked();
    return -1;
}

static rm_u32 rm_push_method_header(rm_u32 subchannel, rm_u32 method, rm_u32 count, rm_u32 sec_op)
{
    return ((sec_op & 0x7u) << 29) |
           ((count & 0x1fffu) << 16) |
           ((subchannel & 0x7u) << 13) |
           ((method >> 2) & 0xfffu);
}

static void rm_write_gpfifo_entry(volatile rm_u32 *gp_words, rm_u32 entry,
                                  rm_u64 pushbuffer_va, rm_u32 pushbuffer_bytes)
{
    gp_words[entry * 2U] = (rm_u32)(pushbuffer_va & 0xfffffffcu);
    gp_words[entry * 2U + 1U] = (rm_u32)(((pushbuffer_va >> 32) & 0xffu) |
                                          (((rm_u64)(pushbuffer_bytes / 4U) & 0x1fffffu) << 10));
}

static int rm_compute_init_locked(void)
{
    static const rm_u32 classes[] = {
        RM_BLACKWELL_COMPUTE_B,
        RM_BLACKWELL_COMPUTE_A,
        RM_HOPPER_COMPUTE_A,
        RM_AMPERE_COMPUTE_B,
        RM_AMPERE_COMPUTE_A,
    };

    if (g.rm_compute != 0 && g.rm_compute_class_engine_id != 0) {
        return 0;
    }
    if (rm_channel_init_locked() != 0) {
        return -1;
    }

    for (size_t i = 0; i < sizeof(classes) / sizeof(classes[0]); i++) {
        rm_handle compute = g.next_rm_handle++;
        if (rm_alloc_object_locked(g.rm_client, g.rm_channel, &compute, classes[i],
                                   NULL, 0, "rm_alloc_compute") != 0) {
            continue;
        }

        rm_get_class_engine_id_params_t params;
        memset(&params, 0, sizeof(params));
        params.hObject = compute;
        if (rm_control_locked(g.rm_channel, RM_NV906F_CTRL_GET_CLASS_ENGINEID,
                              &params, sizeof(params), "rm_compute_class_engine_id") != 0 ||
            params.classEngineID == 0) {
            rm_free_object_locked(g.rm_client, g.rm_channel, compute);
            continue;
        }

        g.rm_compute = compute;
        g.rm_compute_class = classes[i];
        g.rm_compute_class_engine_id = params.classEngineID;
        tracef("RM compute object ready object=0x%x class=0x%x classEngineID=0x%08x engineID=0x%08x",
               g.rm_compute, g.rm_compute_class, g.rm_compute_class_engine_id, params.engineID);
        return 0;
    }

    tracef("RM compute object allocation failed");
    return -1;
}

static int rm_submit_noop_locked(void)
{
    if (rm_channel_init_locked() != 0) {
        return -1;
    }
    volatile rm_u32 *gp_words = (volatile rm_u32 *)g.rm_gpfifo_cpu;
    volatile rm_u32 *control = (volatile rm_u32 *)g.rm_userd_cpu;
    rm_u32 entry = g.rm_gpfifo_put % 32U;
    gp_words[entry * 2U] = 0;
    gp_words[entry * 2U + 1U] = 0;
    __sync_synchronize();
    g.rm_gpfifo_put = (entry + 1U) % 32U;
    control[0x8c / 4] = g.rm_gpfifo_put;
    __sync_synchronize();
    tracef("RM submitted GPFIFO NOP channel=0x%x put=%u token=0x%08x",
           g.rm_channel, g.rm_gpfifo_put, g.rm_work_submit_token);
    return 0;
}

static int rm_submit_compute_set_object_locked(void)
{
    if (rm_compute_init_locked() != 0) {
        return -1;
    }

    volatile rm_u32 *gp_words = (volatile rm_u32 *)g.rm_gpfifo_cpu;
    volatile rm_u32 *control = (volatile rm_u32 *)g.rm_userd_cpu;
    rm_u32 entry = g.rm_gpfifo_put % 32U;
    rm_u32 pb_offset = 0x200u + (entry * 0x40u);
    volatile rm_u32 *pb_words = (volatile rm_u32 *)((volatile uint8_t *)g.rm_gpfifo_cpu + pb_offset);

    pb_words[0] = rm_push_method_header(RM_NVA06F_SUBCHANNEL_COMPUTE, RM_COMPUTE_SET_OBJECT,
                                        1, RM_NVC46F_DMA_SEC_OP_INC_METHOD);
    pb_words[1] = g.rm_compute_class_engine_id;
    pb_words[2] = rm_push_method_header(RM_NVA06F_SUBCHANNEL_COMPUTE, RM_COMPUTE_NO_OPERATION,
                                        1, RM_NVC46F_DMA_SEC_OP_INC_METHOD);
    pb_words[3] = 0;
    pb_words[4] = rm_push_method_header(RM_NVA06F_SUBCHANNEL_COMPUTE, RM_COMPUTE_PIPE_NOP,
                                        1, RM_NVC46F_DMA_SEC_OP_INC_METHOD);
    pb_words[5] = 0;

    rm_write_gpfifo_entry(gp_words, entry, g.rm_gpfifo_va + pb_offset, 6u * sizeof(rm_u32));
    __sync_synchronize();
    g.rm_gpfifo_put = (entry + 1U) % 32U;
    control[0x8c / 4] = g.rm_gpfifo_put;
    __sync_synchronize();
    tracef("RM submitted compute SET_OBJECT PB channel=0x%x compute=0x%x class=0x%x put=%u pb=0x%llx token=0x%08x",
           g.rm_channel, g.rm_compute, g.rm_compute_class, g.rm_gpfifo_put,
           (unsigned long long)(g.rm_gpfifo_va + pb_offset), g.rm_work_submit_token);
    return 0;
}

static int rm_alloc_mapped_system_locked(size_t size, struct allocation *a)
{
    if (rm_init_locked() != 0 || !g.rm_ready) {
        return -1;
    }

    size_t mapped_size = page_align_size(size);
    rm_handle hMemory = g.next_rm_handle++;
    rm_u32 flags = RM_NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS |
                   RM_NVOS02_FLAGS_LOCATION_PCI |
                   RM_NVOS02_FLAGS_COHERENCY_WRITE_COMBINE |
                   RM_NVOS02_FLAGS_GPU_CACHEABLE_YES |
                   RM_NVOS02_FLAGS_MAPPING_NO_MAP;

    rm_nvos02_with_fd_t alloc_api;
    memset(&alloc_api, 0, sizeof(alloc_api));
    alloc_api.params.hRoot = g.rm_client;
    alloc_api.params.hObjectParent = g.rm_device;
    alloc_api.params.hObjectNew = hMemory;
    alloc_api.params.hClass = RM_NV01_MEMORY_SYSTEM;
    alloc_api.params.flags = (rm_v32)flags;
    alloc_api.params.limit = mapped_size - 1U;
    alloc_api.fd = g.ctl_fd;
    int rc = rm_ioctl_xfer(g.gpu_fd, RM_ESC_RM_ALLOC_MEMORY, &alloc_api, sizeof(alloc_api));
    if (rc != 0 || alloc_api.params.status != RM_NV_OK) {
        tracef("RM alloc system memory failed ioctl=%d errno=%d status=0x%08x",
               rc, rc == 0 ? 0 : errno, (rm_u32)alloc_api.params.status);
        return -1;
    }

    rm_nvos33_with_fd_t map_api;
    memset(&map_api, 0, sizeof(map_api));
    map_api.params.hClient = g.rm_client;
    map_api.params.hDevice = g.rm_device;
    map_api.params.hMemory = hMemory;
    map_api.params.length = mapped_size;
    map_api.fd = g.ctl_fd;
    rc = rm_ioctl_xfer(g.ctl_fd, RM_ESC_RM_MAP_MEMORY, &map_api, sizeof(map_api));
    if (rc != 0 || map_api.params.status != RM_NV_OK) {
        tracef("RM map system memory failed ioctl=%d errno=%d status=0x%08x",
               rc, rc == 0 ? 0 : errno, map_api.params.status);
        rm_free_object_locked(g.rm_client, g.rm_device, hMemory);
        return -1;
    }

    void *addr = mmap(NULL, mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED, g.ctl_fd, 0);
    if (addr == MAP_FAILED) {
        tracef("RM mmap system memory failed errno=%d", errno);
        rm_nvos34_t unmap_api;
        memset(&unmap_api, 0, sizeof(unmap_api));
        unmap_api.hClient = g.rm_client;
        unmap_api.hDevice = g.rm_device;
        unmap_api.hMemory = hMemory;
        unmap_api.pLinearAddress = map_api.params.pLinearAddress;
        rm_ioctl_xfer(g.ctl_fd, RM_ESC_RM_UNMAP_MEMORY, &unmap_api, sizeof(unmap_api));
        rm_free_object_locked(g.rm_client, g.rm_device, hMemory);
        return -1;
    }

    memset(addr, 0, mapped_size);
    a->dptr = (CUdeviceptr)(uintptr_t)addr;
    a->host = addr;
    a->size = size;
    a->mapped_size = mapped_size;
    a->memory_type = CU_MEMORYTYPE_DEVICE;
    a->owns_host = false;
    a->rm_backed = true;
    a->rm_memory = hMemory;
    a->rm_linear = map_api.params.pLinearAddress;
    tracef("RM alloc+map system memory %zu/%zu -> handle=0x%x addr=%p linear=0x%llx",
           size, mapped_size, hMemory, addr, (unsigned long long)a->rm_linear);
    return 0;
}

static void rm_release_allocation_locked(struct allocation *a)
{
    if (a == NULL || !a->rm_backed) {
        return;
    }
    if (a->host != NULL && a->mapped_size != 0) {
        munmap(a->host, a->mapped_size);
    }
    if (g.rm_ready && a->rm_memory != 0) {
        rm_nvos34_t unmap_api;
        memset(&unmap_api, 0, sizeof(unmap_api));
        unmap_api.hClient = g.rm_client;
        unmap_api.hDevice = g.rm_device;
        unmap_api.hMemory = a->rm_memory;
        unmap_api.pLinearAddress = a->rm_linear;
        int rc = rm_ioctl_xfer(g.ctl_fd, RM_ESC_RM_UNMAP_MEMORY, &unmap_api, sizeof(unmap_api));
        tracef("RM unmap memory handle=0x%x ioctl=%d errno=%d status=0x%08x",
               a->rm_memory, rc, rc == 0 ? 0 : errno, unmap_api.status);
        rm_free_object_locked(g.rm_client, g.rm_device, a->rm_memory);
    }
}

static char *read_text_file(const char *path)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0 || st.st_size > 1024 * 1024) {
        close(fd);
        return NULL;
    }
    size_t size = st.st_size == 0 ? 4096 : (size_t)st.st_size;
    char *buf = calloc(size + 1, 1);
    if (buf == NULL) {
        close(fd);
        return NULL;
    }
    ssize_t got = read(fd, buf, size);
    close(fd);
    if (got < 0) {
        free(buf);
        return NULL;
    }
    buf[got] = '\0';
    return buf;
}

static void copy_trimmed(char *dst, size_t dst_size, const char *src)
{
    while (*src != '\0' && isspace((unsigned char)*src)) {
        src++;
    }
    size_t len = strlen(src);
    while (len > 0 && isspace((unsigned char)src[len - 1])) {
        len--;
    }
    if (dst_size == 0) {
        return;
    }
    if (len >= dst_size) {
        len = dst_size - 1;
    }
    memcpy(dst, src, len);
    dst[len] = '\0';
}

static void parse_info_line(char *line)
{
    char *colon = strchr(line, ':');
    if (colon == NULL) {
        return;
    }
    *colon = '\0';
    char *key = line;
    char *value = colon + 1;
    if (strstr(key, "Model") != NULL) {
        copy_trimmed(g.name, sizeof(g.name), value);
    } else if (strstr(key, "Bus Location") != NULL) {
        copy_trimmed(g.bus_id, sizeof(g.bus_id), value);
    } else if (strstr(key, "GPU UUID") != NULL) {
        copy_trimmed(g.uuid_text, sizeof(g.uuid_text), value);
    }
}

static void parse_gpu_information(void)
{
    char *text = read_text_file("/proc/driver/nvidia/gpus/0001:01:00.0/information");
    if (text == NULL) {
        text = read_text_file("/proc/driver/nvidia/gpus/0000:01:00.0/information");
    }
    if (text != NULL) {
        char *save = NULL;
        for (char *line = strtok_r(text, "\n", &save); line != NULL; line = strtok_r(NULL, "\n", &save)) {
            parse_info_line(line);
        }
        free(text);
    }

    memset(&g.uuid, 0, sizeof(g.uuid));
    if (strncmp(g.uuid_text, "GPU-", 4) == 0) {
        const char *p = g.uuid_text + 4;
        int byte_idx = 0;
        while (*p != '\0' && byte_idx < 16) {
            while (*p == '-') {
                p++;
            }
            if (!isxdigit((unsigned char)p[0]) || !isxdigit((unsigned char)p[1])) {
                break;
            }
            char tmp[3] = {p[0], p[1], '\0'};
            g.uuid.bytes[byte_idx++] = (char)strtoul(tmp, NULL, 16);
            p += 2;
        }
    }
}

static void configure_device_defaults(void)
{
    unsigned long long total_mb = env_ull("LANXIN_NVIDIA_CUDA_TOTAL_MEM_MB", DEFAULT_TOTAL_MEM_MB);
    g.total_mem = total_mb * 1024ULL * 1024ULL;
    g.sm_count = (int)env_ull("LANXIN_NVIDIA_CUDA_SM_COUNT", DEFAULT_SM_COUNT);
    g.cc_major = (int)env_ull("LANXIN_NVIDIA_CUDA_CC_MAJOR", DEFAULT_COMPUTE_MAJOR);
    g.cc_minor = (int)env_ull("LANXIN_NVIDIA_CUDA_CC_MINOR", DEFAULT_COMPUTE_MINOR);
}

static CUresult ensure_initialized_locked(void)
{
    if (g.initialized) {
        return g.init_result;
    }

    configure_device_defaults();
    parse_gpu_information();

    g.ctl_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    g.gpu_fd = open("/dev/nvidia0", O_RDWR | O_CLOEXEC);
    g.uvm_fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    g.uvm_tools_fd = open("/dev/nvidia-uvm-tools", O_RDWR | O_CLOEXEC);

    if (g.ctl_fd < 0 || g.gpu_fd < 0) {
        tracef("failed to open NVIDIA nodes: ctl=%d gpu=%d errno=%d", g.ctl_fd, g.gpu_fd, errno);
        g.device_count = 0;
        g.init_result = CUDA_ERROR_NO_DEVICE;
    } else {
        g.device_count = 1;
        g.init_result = CUDA_SUCCESS;
        tracef("opened NVIDIA kernel driver ctl=%d gpu=%d uvm=%d", g.ctl_fd, g.gpu_fd, g.uvm_fd);
        (void)rm_init_locked();
    }
    g.initialized = true;
    return g.init_result;
}

static CUresult ensure_initialized(void)
{
    pthread_mutex_lock(&g_lock);
    CUresult result = ensure_initialized_locked();
    pthread_mutex_unlock(&g_lock);
    return result;
}

static bool valid_device(CUdevice dev)
{
    return dev == 0 && g.device_count > 0;
}

static CUcontext make_context_locked(CUdevice dev, unsigned int flags, bool primary)
{
    CUcontext ctx = calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        return NULL;
    }
    ctx->magic = HANDLE_MAGIC_CTX;
    ctx->dev = dev;
    ctx->flags = flags;
    ctx->id = g.next_id++;
    ctx->primary = primary;
    return ctx;
}

static bool valid_ctx(CUcontext ctx)
{
    return ctx != NULL && ctx->magic == HANDLE_MAGIC_CTX;
}

static bool valid_stream(CUstream stream)
{
    return stream == NULL || stream->magic == HANDLE_MAGIC_STREAM;
}

static bool valid_event(CUevent event)
{
    return event != NULL && event->magic == HANDLE_MAGIC_EVENT;
}

static bool valid_module(CUmodule module)
{
    return module != NULL && module->magic == HANDLE_MAGIC_MODULE;
}

static bool valid_function(CUfunction function)
{
    return function != NULL && function->magic == HANDLE_MAGIC_FUNC;
}

static bool valid_link_state(CUlinkState state)
{
    return state != NULL && state->magic == HANDLE_MAGIC_LINK;
}

static bool valid_library(CUlibrary library)
{
    return library != NULL && library->magic == HANDLE_MAGIC_LIBRARY;
}

static bool valid_kernel(CUkernel kernel)
{
    return kernel != NULL && kernel->magic == HANDLE_MAGIC_KERNEL;
}

static CUcontext current_or_primary_locked(void)
{
    if (valid_ctx(tls_current_ctx)) {
        return tls_current_ctx;
    }
    if (g.primary_ctx == NULL) {
        g.primary_ctx = make_context_locked(0, g.primary_flags, true);
    }
    return g.primary_ctx;
}

static char *read_file_bytes(const char *path, size_t *size_out)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    size_t size = (size_t)len;
    char *buf = calloc(1, size + 1U);
    if (buf == NULL) {
        fclose(fp);
        return NULL;
    }
    if (size != 0 && fread(buf, 1, size, fp) != size) {
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    if (size_out != NULL) {
        *size_out = size;
    }
    return buf;
}

static bool range_contains(CUdeviceptr base, size_t size, CUdeviceptr ptr, size_t bytes)
{
    if (ptr < base) {
        return false;
    }
    uint64_t offset = ptr - base;
    if (offset > size) {
        return false;
    }
    return bytes <= size - (size_t)offset;
}

static struct allocation *find_alloc_locked(CUdeviceptr ptr, size_t bytes)
{
    for (struct allocation *a = g.allocs; a != NULL; a = a->next) {
        if (range_contains(a->dptr, a->size, ptr, bytes)) {
            return a;
        }
    }
    return NULL;
}

static void *alloc_host_memory(size_t bytes)
{
    void *ptr = NULL;
    size_t actual = bytes == 0 ? 1 : bytes;
    if (posix_memalign(&ptr, 256, actual) != 0) {
        return NULL;
    }
    memset(ptr, 0, actual);
    return ptr;
}

static CUresult add_allocation_locked(CUdeviceptr *dptr, void *host, size_t size, unsigned int memory_type, bool owns_host)
{
    struct allocation *a = calloc(1, sizeof(*a));
    if (a == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    a->dptr = (CUdeviceptr)(uintptr_t)host;
    a->host = host;
    a->size = size;
    a->mapped_size = size;
    a->memory_type = memory_type;
    a->owns_host = owns_host;
    a->id = g.next_id++;
    a->next = g.allocs;
    g.allocs = a;
    if (memory_type == CU_MEMORYTYPE_DEVICE || memory_type == CU_MEMORYTYPE_UNIFIED) {
        g.allocated_bytes += size;
    }
    *dptr = a->dptr;
    return CUDA_SUCCESS;
}

static CUresult resolve_device_ptr_locked(CUdeviceptr dptr, size_t bytes, void **host_out)
{
    struct allocation *a = find_alloc_locked(dptr, bytes);
    if (a == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *host_out = (char *)a->host + (dptr - a->dptr);
    return CUDA_SUCCESS;
}

static void *resolve_uva_ptr_locked(CUdeviceptr ptr, size_t bytes)
{
    struct allocation *a = find_alloc_locked(ptr, bytes);
    if (a != NULL) {
        return (char *)a->host + (ptr - a->dptr);
    }
    return (void *)(uintptr_t)ptr;
}

static CUresult remove_allocation_locked(CUdeviceptr dptr, bool free_owned)
{
    struct allocation **prev = &g.allocs;
    while (*prev != NULL) {
        struct allocation *a = *prev;
        if (a->dptr == dptr) {
            *prev = a->next;
            if (a->memory_type == CU_MEMORYTYPE_DEVICE || a->memory_type == CU_MEMORYTYPE_UNIFIED) {
                g.allocated_bytes -= a->size;
            }
            if (a->rm_backed) {
                rm_release_allocation_locked(a);
            }
            if (free_owned && a->owns_host) {
                free(a->host);
            }
            free(a);
            return CUDA_SUCCESS;
        }
        prev = &a->next;
    }
    return CUDA_ERROR_INVALID_VALUE;
}

static CUresult set_attr_value(void *data, CUpointer_attribute attribute, struct allocation *a)
{
    if (data == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    switch (attribute) {
    case CU_POINTER_ATTRIBUTE_CONTEXT:
        *(CUcontext *)data = tls_current_ctx;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
        *(unsigned int *)data = a != NULL ? a->memory_type : 0;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
        *(CUdeviceptr *)data = a != NULL ? a->dptr : 0;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_HOST_POINTER:
        *(void **)data = a != NULL ? a->host : NULL;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_BUFFER_ID:
        *(unsigned long long *)data = a != NULL ? a->id : 0;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_IS_MANAGED:
        *(int *)data = a != NULL && a->memory_type == CU_MEMORYTYPE_UNIFIED;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
        *(int *)data = a != NULL ? 0 : -1;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
        *(CUdeviceptr *)data = a != NULL ? a->dptr : 0;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
        *(size_t *)data = a != NULL ? a->size : 0;
        return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_MAPPED:
        *(int *)data = a != NULL;
        return CUDA_SUCCESS;
    default:
        return CUDA_ERROR_NOT_SUPPORTED;
    }
}

CUresult CUDAAPI cuInit(unsigned int Flags)
{
    if (Flags != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return ensure_initialized();
}

CUresult CUDAAPI cuDriverGetVersion(int *driverVersion)
{
    if (driverVersion == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *driverVersion = LANXIN_CUDA_VERSION;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetCount(int *count)
{
    if (count == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = ensure_initialized();
    pthread_mutex_lock(&g_lock);
    *count = result == CUDA_SUCCESS ? g.device_count : 0;
    pthread_mutex_unlock(&g_lock);
    return result == CUDA_ERROR_NO_DEVICE ? CUDA_SUCCESS : result;
}

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal)
{
    if (device == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = ensure_initialized();
    if (result != CUDA_SUCCESS) {
        return result;
    }
    pthread_mutex_lock(&g_lock);
    bool ok = ordinal >= 0 && ordinal < g.device_count;
    pthread_mutex_unlock(&g_lock);
    if (!ok) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev)
{
    if (name == NULL || len <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = ensure_initialized();
    if (result != CUDA_SUCCESS) {
        return result;
    }
    pthread_mutex_lock(&g_lock);
    if (!valid_device(dev)) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_INVALID_DEVICE;
    }
    snprintf(name, (size_t)len, "%s", g.name);
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev)
{
    if (uuid == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = ensure_initialized();
    if (result != CUDA_SUCCESS) {
        return result;
    }
    pthread_mutex_lock(&g_lock);
    if (!valid_device(dev)) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *uuid = g.uuid;
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev)
{
    return cuDeviceGetUuid_v2(uuid, dev);
}

CUresult CUDAAPI cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev)
{
    if (bytes == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = ensure_initialized();
    if (result != CUDA_SUCCESS) {
        return result;
    }
    pthread_mutex_lock(&g_lock);
    if (!valid_device(dev)) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *bytes = g.total_mem;
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev)
{
    return cuDeviceTotalMem_v2(bytes, dev);
}

CUresult CUDAAPI cuDeviceComputeCapability(int *major, int *minor, CUdevice dev)
{
    if (major == NULL || minor == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = ensure_initialized();
    if (result != CUDA_SUCCESS) {
        return result;
    }
    pthread_mutex_lock(&g_lock);
    if (!valid_device(dev)) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *major = g.cc_major;
    *minor = g.cc_minor;
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev)
{
    if (pciBusId == NULL || len <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = ensure_initialized();
    if (result != CUDA_SUCCESS) {
        return result;
    }
    pthread_mutex_lock(&g_lock);
    if (!valid_device(dev)) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_INVALID_DEVICE;
    }
    snprintf(pciBusId, (size_t)len, "%s", g.bus_id);
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId)
{
    if (dev == NULL || pciBusId == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = ensure_initialized();
    if (result != CUDA_SUCCESS) {
        return result;
    }
    pthread_mutex_lock(&g_lock);
    bool ok = g.device_count > 0 && strcmp(pciBusId, g.bus_id) == 0;
    pthread_mutex_unlock(&g_lock);
    if (!ok) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *dev = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
    if (pi == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = ensure_initialized();
    if (result != CUDA_SUCCESS) {
        return result;
    }
    pthread_mutex_lock(&g_lock);
    if (!valid_device(dev)) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_INVALID_DEVICE;
    }
    int value = 0;
    switch (attrib) {
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: value = 1024; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: value = 1024; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: value = 1024; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: value = 64; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: value = 2147483647; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: value = 65535; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: value = 65535; break;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: value = 49152; break;
    case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: value = 65536; break;
    case CU_DEVICE_ATTRIBUTE_WARP_SIZE: value = 32; break;
    case CU_DEVICE_ATTRIBUTE_MAX_PITCH: value = 2147483647; break;
    case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: value = 65536; break;
    case CU_DEVICE_ATTRIBUTE_CLOCK_RATE: value = 2500000; break;
    case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: value = 512; break;
    case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP: value = 1; break;
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: value = g.sm_count; break;
    case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT: value = 0; break;
    case CU_DEVICE_ATTRIBUTE_INTEGRATED: value = 0; break;
    case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: value = 1; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE: value = CU_COMPUTEMODE_DEFAULT; break;
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: value = 1; break;
    case CU_DEVICE_ATTRIBUTE_ECC_ENABLED: value = 0; break;
    case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: value = 1; break;
    case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: value = 0; break;
    case CU_DEVICE_ATTRIBUTE_TCC_DRIVER: value = 0; break;
    case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: value = 1400000; break;
    case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: value = 512; break;
    case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: value = 98304 * 1024; break;
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: value = 1536; break;
    case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT: value = 2; break;
    case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: value = 1; break;
    case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: value = 1; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: value = g.cc_major; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: value = g.cc_minor; break;
    case CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED: value = 1; break;
    case CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED: value = 1; break;
    case CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED: value = 1; break;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: value = 102400; break;
    case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR: value = 65536; break;
    case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY: value = 0; break;
    case CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD: value = 0; break;
    case CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED: value = 0; break;
    case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS: value = 1; break;
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS: value = 0; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED: value = 1; break;
    case CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM: value = 1; break;
    case CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED: value = 1; break;
    case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES: value = 1; break;
    case CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST: value = 1; break;
    case CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED: value = 0; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR: value = 32; break;
    case CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED: value = 0; break;
    case CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH: value = 0; break;
    case CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS: value = 0; break;
    case CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID: value = 0x2b8710de; break;
    case CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID: value = -1; break;
    default:
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    *pi = value;
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    if (pctx == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = ensure_initialized();
    if (result != CUDA_SUCCESS) {
        return result;
    }
    pthread_mutex_lock(&g_lock);
    if (!valid_device(dev)) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_INVALID_DEVICE;
    }
    CUcontext ctx = make_context_locked(dev, flags, false);
    if (ctx == NULL) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *pctx = ctx;
    tls_current_ctx = ctx;
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    return cuCtxCreate_v2(pctx, flags, dev);
}

CUresult CUDAAPI cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *paramsArray, int numParams, unsigned int flags, CUdevice dev)
{
    (void)paramsArray;
    (void)numParams;
    return cuCtxCreate_v2(pctx, flags, dev);
}

CUresult CUDAAPI cuCtxDestroy_v2(CUcontext ctx)
{
    if (!valid_ctx(ctx)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (ctx->primary) {
        return CUDA_SUCCESS;
    }
    if (tls_current_ctx == ctx) {
        tls_current_ctx = NULL;
    }
    ctx->magic = 0;
    free(ctx);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDestroy(CUcontext ctx)
{
    return cuCtxDestroy_v2(ctx);
}

CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx)
{
    if (ctx != NULL && !valid_ctx(ctx)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    tls_current_ctx = ctx;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx)
{
    if (pctx == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pctx = tls_current_ctx;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx)
{
    if (!valid_ctx(ctx)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (tls_ctx_depth >= (int)(sizeof(tls_ctx_stack) / sizeof(tls_ctx_stack[0]))) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    tls_ctx_stack[tls_ctx_depth++] = tls_current_ctx;
    tls_current_ctx = ctx;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx)
{
    if (pctx == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pctx = tls_current_ctx;
    tls_current_ctx = tls_ctx_depth > 0 ? tls_ctx_stack[--tls_ctx_depth] : NULL;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetDevice(CUdevice *device)
{
    if (device == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    pthread_mutex_lock(&g_lock);
    CUcontext ctx = current_or_primary_locked();
    if (!valid_ctx(ctx)) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    *device = ctx->dev;
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetFlags(unsigned int *flags)
{
    if (flags == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    pthread_mutex_lock(&g_lock);
    CUcontext ctx = current_or_primary_locked();
    *flags = valid_ctx(ctx) ? ctx->flags : 0;
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSynchronize(void)
{
    return ensure_initialized();
}

CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int *version)
{
    if (version == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (ctx != NULL && !valid_ctx(ctx)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    *version = LANXIN_CUDA_VERSION;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetId(CUcontext ctx, unsigned long long *ctxId)
{
    if (ctxId == NULL || !valid_ctx(ctx)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *ctxId = ctx->id;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev)
{
    if (pctx == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = ensure_initialized();
    if (result != CUDA_SUCCESS) {
        return result;
    }
    pthread_mutex_lock(&g_lock);
    if (!valid_device(dev)) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if (g.primary_ctx == NULL) {
        g.primary_ctx = make_context_locked(dev, g.primary_flags, true);
    }
    if (g.primary_ctx == NULL) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *pctx = g.primary_ctx;
    tls_current_ctx = g.primary_ctx;
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxRelease_v2(CUdevice dev)
{
    (void)dev;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev)
{
    return cuDevicePrimaryCtxRelease_v2(dev);
}

CUresult CUDAAPI cuDevicePrimaryCtxReset_v2(CUdevice dev)
{
    (void)dev;
    pthread_mutex_lock(&g_lock);
    if (g.primary_ctx != NULL) {
        if (tls_current_ctx == g.primary_ctx) {
            tls_current_ctx = NULL;
        }
        g.primary_ctx->magic = 0;
        free(g.primary_ctx);
        g.primary_ctx = NULL;
    }
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev)
{
    return cuDevicePrimaryCtxReset_v2(dev);
}

CUresult CUDAAPI cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags)
{
    (void)dev;
    pthread_mutex_lock(&g_lock);
    g.primary_flags = flags;
    if (g.primary_ctx != NULL) {
        g.primary_ctx->flags = flags;
    }
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags)
{
    return cuDevicePrimaryCtxSetFlags_v2(dev, flags);
}

CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active)
{
    (void)dev;
    if (flags == NULL || active == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    pthread_mutex_lock(&g_lock);
    *flags = g.primary_flags;
    *active = g.primary_ctx != NULL;
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetInfo_v2(size_t *free_bytes, size_t *total_bytes)
{
    if (free_bytes == NULL || total_bytes == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = ensure_initialized();
    if (result != CUDA_SUCCESS) {
        return result;
    }
    pthread_mutex_lock(&g_lock);
    *total_bytes = g.total_mem;
    *free_bytes = g.allocated_bytes < g.total_mem ? g.total_mem - g.allocated_bytes : 0;
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetInfo(size_t *free_bytes, size_t *total_bytes)
{
    return cuMemGetInfo_v2(free_bytes, total_bytes);
}

CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
{
    if (dptr == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = ensure_initialized();
    if (result != CUDA_SUCCESS) {
        return result;
    }
    struct allocation *rm_alloc = calloc(1, sizeof(*rm_alloc));
    if (rm_alloc != NULL) {
        pthread_mutex_lock(&g_lock);
        if (rm_alloc_mapped_system_locked(bytesize, rm_alloc) == 0) {
            rm_alloc->id = g.next_id++;
            rm_alloc->next = g.allocs;
            g.allocs = rm_alloc;
            g.allocated_bytes += bytesize;
            *dptr = rm_alloc->dptr;
            pthread_mutex_unlock(&g_lock);
            tracef("cuMemAlloc_v2 RM-backed %zu -> 0x%llx", bytesize, (unsigned long long)*dptr);
            return CUDA_SUCCESS;
        }
        pthread_mutex_unlock(&g_lock);
        free(rm_alloc);
    }
    void *host = alloc_host_memory(bytesize);
    if (host == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    pthread_mutex_lock(&g_lock);
    result = add_allocation_locked(dptr, host, bytesize, CU_MEMORYTYPE_DEVICE, true);
    pthread_mutex_unlock(&g_lock);
    if (result != CUDA_SUCCESS) {
        free(host);
    }
    tracef("cuMemAlloc_v2 %zu -> 0x%llx", bytesize, (unsigned long long)(dptr != NULL ? *dptr : 0));
    return result;
}

CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize)
{
    return cuMemAlloc_v2(dptr, bytesize);
}

CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags)
{
    (void)flags;
    if (dptr == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = ensure_initialized();
    if (result != CUDA_SUCCESS) {
        return result;
    }
    void *host = alloc_host_memory(bytesize);
    if (host == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    pthread_mutex_lock(&g_lock);
    result = add_allocation_locked(dptr, host, bytesize, CU_MEMORYTYPE_UNIFIED, true);
    pthread_mutex_unlock(&g_lock);
    if (result != CUDA_SUCCESS) {
        free(host);
    }
    return result;
}

CUresult CUDAAPI cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes)
{
    (void)ElementSizeBytes;
    if (dptr == NULL || pPitch == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    size_t pitch = (WidthInBytes + 127U) & ~(size_t)127U;
    if (Height != 0 && pitch > SIZE_MAX / Height) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *pPitch = pitch;
    return cuMemAlloc_v2(dptr, pitch * Height);
}

CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes)
{
    return cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

CUresult CUDAAPI cuMemFree_v2(CUdeviceptr dptr)
{
    pthread_mutex_lock(&g_lock);
    CUresult result = remove_allocation_locked(dptr, true);
    pthread_mutex_unlock(&g_lock);
    tracef("cuMemFree_v2 0x%llx -> %d", (unsigned long long)dptr, result);
    return result;
}

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr)
{
    return cuMemFree_v2(dptr);
}

CUresult CUDAAPI cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr)
{
    pthread_mutex_lock(&g_lock);
    struct allocation *a = find_alloc_locked(dptr, 0);
    if (a == NULL) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (pbase != NULL) {
        *pbase = a->dptr;
    }
    if (psize != NULL) {
        *psize = a->size;
    }
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr)
{
    return cuMemGetAddressRange_v2(pbase, psize, dptr);
}

CUresult CUDAAPI cuMemAllocHost_v2(void **pp, size_t bytesize)
{
    if (pp == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    void *host = alloc_host_memory(bytesize);
    if (host == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUdeviceptr dptr = 0;
    pthread_mutex_lock(&g_lock);
    CUresult result = add_allocation_locked(&dptr, host, bytesize, CU_MEMORYTYPE_HOST, true);
    pthread_mutex_unlock(&g_lock);
    if (result != CUDA_SUCCESS) {
        free(host);
        return result;
    }
    *pp = host;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAllocHost(void **pp, size_t bytesize)
{
    return cuMemAllocHost_v2(pp, bytesize);
}

CUresult CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags)
{
    (void)Flags;
    return cuMemAllocHost_v2(pp, bytesize);
}

CUresult CUDAAPI cuMemFreeHost(void *p)
{
    if (p == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    pthread_mutex_lock(&g_lock);
    CUresult result = remove_allocation_locked((CUdeviceptr)(uintptr_t)p, true);
    pthread_mutex_unlock(&g_lock);
    return result;
}

CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags)
{
    (void)Flags;
    if (p == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUdeviceptr dptr = 0;
    pthread_mutex_lock(&g_lock);
    CUresult result = add_allocation_locked(&dptr, p, bytesize, CU_MEMORYTYPE_HOST, false);
    pthread_mutex_unlock(&g_lock);
    return result;
}

CUresult CUDAAPI cuMemHostUnregister(void *p)
{
    if (p == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    pthread_mutex_lock(&g_lock);
    CUresult result = remove_allocation_locked((CUdeviceptr)(uintptr_t)p, false);
    pthread_mutex_unlock(&g_lock);
    return result;
}

CUresult CUDAAPI cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags)
{
    (void)Flags;
    if (pdptr == NULL || p == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    pthread_mutex_lock(&g_lock);
    struct allocation *a = find_alloc_locked((CUdeviceptr)(uintptr_t)p, 0);
    if (a == NULL) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pdptr = (CUdeviceptr)(uintptr_t)p;
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags)
{
    return cuMemHostGetDevicePointer_v2(pdptr, p, Flags);
}

CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
{
    if (srcHost == NULL && ByteCount != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    pthread_mutex_lock(&g_lock);
    void *dst = NULL;
    CUresult result = resolve_device_ptr_locked(dstDevice, ByteCount, &dst);
    if (result == CUDA_SUCCESS && ByteCount != 0) {
        memcpy(dst, srcHost, ByteCount);
    }
    pthread_mutex_unlock(&g_lock);
    return result;
}

CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
{
    return cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    if (dstHost == NULL && ByteCount != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    pthread_mutex_lock(&g_lock);
    void *src = NULL;
    CUresult result = resolve_device_ptr_locked(srcDevice, ByteCount, &src);
    if (result == CUDA_SUCCESS && ByteCount != 0) {
        memcpy(dstHost, src, ByteCount);
    }
    pthread_mutex_unlock(&g_lock);
    return result;
}

CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    return cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
{
    pthread_mutex_lock(&g_lock);
    void *dst = NULL;
    void *src = NULL;
    CUresult result = resolve_device_ptr_locked(dstDevice, ByteCount, &dst);
    if (result == CUDA_SUCCESS) {
        result = resolve_device_ptr_locked(srcDevice, ByteCount, &src);
    }
    if (result == CUDA_SUCCESS && ByteCount != 0) {
        memmove(dst, src, ByteCount);
    }
    pthread_mutex_unlock(&g_lock);
    return result;
}

CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
{
    return cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);
}

CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
{
    pthread_mutex_lock(&g_lock);
    void *dstp = resolve_uva_ptr_locked(dst, ByteCount);
    void *srcp = resolve_uva_ptr_locked(src, ByteCount);
    if ((dstp == NULL || srcp == NULL) && ByteCount != 0) {
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_INVALID_VALUE;
    }
    memmove(dstp, srcp, ByteCount);
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
{
    if (!valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    return cuMemcpy(dst, src, ByteCount);
}

CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream)
{
    if (!valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    return cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    if (!valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    return cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
}

CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    if (!valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    return cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);
}

CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream)
{
    return cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    return cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    return cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);
}

CUresult CUDAAPI cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy)
{
    if (pCopy == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (pCopy->srcArray != NULL || pCopy->dstArray != NULL) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    for (size_t row = 0; row < pCopy->Height; row++) {
        const char *src = NULL;
        char *dst = NULL;
        pthread_mutex_lock(&g_lock);
        if (pCopy->srcMemoryType == CU_MEMORYTYPE_HOST) {
            src = (const char *)pCopy->srcHost + (pCopy->srcY + row) * pCopy->srcPitch + pCopy->srcXInBytes;
        } else {
            void *tmp = NULL;
            CUresult r = resolve_device_ptr_locked(pCopy->srcDevice + (pCopy->srcY + row) * pCopy->srcPitch + pCopy->srcXInBytes, pCopy->WidthInBytes, &tmp);
            if (r != CUDA_SUCCESS) {
                pthread_mutex_unlock(&g_lock);
                return r;
            }
            src = tmp;
        }
        if (pCopy->dstMemoryType == CU_MEMORYTYPE_HOST) {
            dst = (char *)pCopy->dstHost + (pCopy->dstY + row) * pCopy->dstPitch + pCopy->dstXInBytes;
        } else {
            void *tmp = NULL;
            CUresult r = resolve_device_ptr_locked(pCopy->dstDevice + (pCopy->dstY + row) * pCopy->dstPitch + pCopy->dstXInBytes, pCopy->WidthInBytes, &tmp);
            if (r != CUDA_SUCCESS) {
                pthread_mutex_unlock(&g_lock);
                return r;
            }
            dst = tmp;
        }
        memmove(dst, src, pCopy->WidthInBytes);
        pthread_mutex_unlock(&g_lock);
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D *pCopy)
{
    return cuMemcpy2D_v2(pCopy);
}

CUresult CUDAAPI cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy)
{
    return cuMemcpy2D_v2(pCopy);
}

CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy)
{
    return cuMemcpy2D_v2(pCopy);
}

CUresult CUDAAPI cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream)
{
    if (!valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    return cuMemcpy2D_v2(pCopy);
}

CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream)
{
    return cuMemcpy2DAsync_v2(pCopy, hStream);
}

CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    pthread_mutex_lock(&g_lock);
    void *dst = NULL;
    CUresult result = resolve_device_ptr_locked(dstDevice, N, &dst);
    if (result == CUDA_SUCCESS) {
        memset(dst, uc, N);
    }
    pthread_mutex_unlock(&g_lock);
    return result;
}

CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    return cuMemsetD8_v2(dstDevice, uc, N);
}

CUresult CUDAAPI cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N)
{
    pthread_mutex_lock(&g_lock);
    uint16_t *dst = NULL;
    CUresult result = resolve_device_ptr_locked(dstDevice, N * sizeof(uint16_t), (void **)&dst);
    if (result == CUDA_SUCCESS) {
        for (size_t i = 0; i < N; i++) {
            dst[i] = us;
        }
    }
    pthread_mutex_unlock(&g_lock);
    return result;
}

CUresult CUDAAPI cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N)
{
    return cuMemsetD16_v2(dstDevice, us, N);
}

CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    pthread_mutex_lock(&g_lock);
    uint32_t *dst = NULL;
    CUresult result = resolve_device_ptr_locked(dstDevice, N * sizeof(uint32_t), (void **)&dst);
    if (result == CUDA_SUCCESS) {
        for (size_t i = 0; i < N; i++) {
            dst[i] = ui;
        }
    }
    pthread_mutex_unlock(&g_lock);
    return result;
}

CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    return cuMemsetD32_v2(dstDevice, ui, N);
}

CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
{
    if (!valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    return cuMemsetD8_v2(dstDevice, uc, N);
}

CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
{
    if (!valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    return cuMemsetD16_v2(dstDevice, us, N);
}

CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
{
    if (!valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    return cuMemsetD32_v2(dstDevice, ui, N);
}

CUresult CUDAAPI cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr)
{
    pthread_mutex_lock(&g_lock);
    struct allocation *a = find_alloc_locked(ptr, 0);
    CUresult result = set_attr_value(data, attribute, a);
    pthread_mutex_unlock(&g_lock);
    return result;
}

CUresult CUDAAPI cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr)
{
    if (attributes == NULL || data == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    pthread_mutex_lock(&g_lock);
    struct allocation *a = find_alloc_locked(ptr, 0);
    for (unsigned int i = 0; i < numAttributes; i++) {
        CUresult r = set_attr_value(data[i], attributes[i], a);
        if (r != CUDA_SUCCESS && r != CUDA_ERROR_NOT_SUPPORTED) {
            pthread_mutex_unlock(&g_lock);
            return r;
        }
    }
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags)
{
    if (phStream == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUstream stream = calloc(1, sizeof(*stream));
    if (stream == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    pthread_mutex_lock(&g_lock);
    stream->magic = HANDLE_MAGIC_STREAM;
    stream->flags = Flags;
    stream->priority = 0;
    stream->id = g.next_id++;
    stream->ctx = current_or_primary_locked();
    pthread_mutex_unlock(&g_lock);
    *phStream = stream;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority)
{
    CUresult result = cuStreamCreate(phStream, flags);
    if (result == CUDA_SUCCESS && *phStream != NULL) {
        (*phStream)->priority = priority;
    }
    return result;
}

CUresult CUDAAPI cuStreamDestroy_v2(CUstream hStream)
{
    if (!valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    if (hStream != NULL) {
        hStream->magic = 0;
        free(hStream);
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamDestroy(CUstream hStream)
{
    return cuStreamDestroy_v2(hStream);
}

CUresult CUDAAPI cuStreamQuery(CUstream hStream)
{
    return valid_stream(hStream) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

CUresult CUDAAPI cuStreamSynchronize(CUstream hStream)
{
    return valid_stream(hStream) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority)
{
    if (priority == NULL || !valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *priority = hStream == NULL ? 0 : hStream->priority;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags)
{
    if (flags == NULL || !valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *flags = hStream == NULL ? 0 : hStream->flags;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetId(CUstream hStream, unsigned long long *streamId)
{
    if (streamId == NULL || !valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *streamId = hStream == NULL ? 0 : hStream->id;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetCtx(CUstream hStream, CUcontext *pctx)
{
    if (pctx == NULL || !valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    pthread_mutex_lock(&g_lock);
    *pctx = hStream == NULL ? current_or_primary_locked() : hStream->ctx;
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamGetDevice(CUstream hStream, CUdevice *device)
{
    if (device == NULL || !valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device = hStream != NULL && valid_ctx(hStream->ctx) ? hStream->ctx->dev : 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int Flags)
{
    if (phEvent == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUevent event = calloc(1, sizeof(*event));
    if (event == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    event->magic = HANDLE_MAGIC_EVENT;
    event->flags = Flags;
    *phEvent = event;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream)
{
    if (!valid_event(hEvent) || !valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    clock_gettime(CLOCK_MONOTONIC, &hEvent->stamp);
    hEvent->recorded = true;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags)
{
    (void)flags;
    return cuEventRecord(hEvent, hStream);
}

CUresult CUDAAPI cuEventQuery(CUevent hEvent)
{
    return valid_event(hEvent) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

CUresult CUDAAPI cuEventSynchronize(CUevent hEvent)
{
    return valid_event(hEvent) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

CUresult CUDAAPI cuEventDestroy_v2(CUevent hEvent)
{
    if (!valid_event(hEvent)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    hEvent->magic = 0;
    free(hEvent);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventDestroy(CUevent hEvent)
{
    return cuEventDestroy_v2(hEvent);
}

CUresult CUDAAPI cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd)
{
    if (pMilliseconds == NULL || !valid_event(hStart) || !valid_event(hEnd)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    long sec = hEnd->stamp.tv_sec - hStart->stamp.tv_sec;
    long nsec = hEnd->stamp.tv_nsec - hStart->stamp.tv_nsec;
    *pMilliseconds = (float)sec * 1000.0f + (float)nsec / 1000000.0f;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuEventElapsedTime_v2(float *pMilliseconds, CUevent hStart, CUevent hEnd)
{
    return cuEventElapsedTime(pMilliseconds, hStart, hEnd);
}

static CUmodule make_module(const char *name, const void *image, size_t image_size, bool copy_image)
{
    CUmodule module = calloc(1, sizeof(*module));
    if (module == NULL) {
        return NULL;
    }
    module->magic = HANDLE_MAGIC_MODULE;
    module->name = strdup(name != NULL ? name : "<memory>");
    module->image = image;
    module->image_size = image_size;
    if (copy_image && image != NULL && image_size != 0) {
        module->owned_image = malloc(image_size + 1U);
        if (module->owned_image == NULL) {
            free(module->name);
            free(module);
            return NULL;
        }
        memcpy(module->owned_image, image, image_size);
        ((char *)module->owned_image)[image_size] = '\0';
        module->image = module->owned_image;
    }
    pthread_mutex_lock(&g_lock);
    module->next = g.modules;
    g.modules = module;
    pthread_mutex_unlock(&g_lock);
    return module;
}

CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname)
{
    if (module == NULL || fname == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (access(fname, R_OK) != 0) {
        return CUDA_ERROR_FILE_NOT_FOUND;
    }
    size_t image_size = 0;
    char *image = read_file_bytes(fname, &image_size);
    if (image == NULL) {
        return CUDA_ERROR_INVALID_IMAGE;
    }
    CUmodule m = make_module(fname, image, image_size, true);
    free(image);
    if (m == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *module = m;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image)
{
    if (module == NULL || image == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    size_t image_size = 0;
    const unsigned char *bytes = (const unsigned char *)image;
    if (bytes[0] == 0x7f && bytes[1] == 'E' && bytes[2] == 'L' && bytes[3] == 'F') {
        image_size = 4;
    } else {
        image_size = strnlen((const char *)image, 16U * 1024U * 1024U);
        if (image_size != 16U * 1024U * 1024U) {
            image_size++;
        } else {
            image_size = 0;
        }
    }
    CUmodule m = make_module("<image>", image, image_size, image_size != 0);
    if (m == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *module = m;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    (void)numOptions;
    (void)options;
    (void)optionValues;
    return cuModuleLoadData(module, image);
}

CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin)
{
    return cuModuleLoadData(module, fatCubin);
}

static CUresult module_get_or_create_global(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name)
{
    if (!valid_module(hmod) || name == NULL || (dptr == NULL && bytes == NULL)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (struct module_global *global = hmod->globals; global != NULL; global = global->next) {
        if (strcmp(global->name, name) == 0) {
            if (dptr != NULL) {
                *dptr = global->dptr;
            }
            if (bytes != NULL) {
                *bytes = global->bytes;
            }
            return CUDA_SUCCESS;
        }
    }

    size_t global_bytes = (size_t)env_ull("LANXIN_NVIDIA_CUDA_GLOBAL_BYTES", 4096);
    if (global_bytes == 0) {
        global_bytes = 1;
    }
    CUdeviceptr ptr = 0;
    CUresult result = cuMemAlloc_v2(&ptr, global_bytes);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    struct module_global *global = calloc(1, sizeof(*global));
    if (global == NULL) {
        cuMemFree_v2(ptr);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    global->name = strdup(name);
    if (global->name == NULL) {
        free(global);
        cuMemFree_v2(ptr);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    global->dptr = ptr;
    global->bytes = global_bytes;
    global->next = hmod->globals;
    hmod->globals = global;
    if (dptr != NULL) {
        *dptr = ptr;
    }
    if (bytes != NULL) {
        *bytes = global_bytes;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name)
{
    return module_get_or_create_global(dptr, bytes, hmod, name);
}

CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name)
{
    return cuModuleGetGlobal_v2(dptr, bytes, hmod, name);
}

static CUresult link_set_image(CUlinkState state, const void *data, size_t size)
{
    if (!valid_link_state(state) || (data == NULL && size != 0)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const char fallback[] = "LANXIN_FAKE_CUDA_LINK_IMAGE";
    const void *src = data != NULL ? data : fallback;
    size_t src_size = data != NULL ? size : sizeof(fallback);
    if (src_size == 0) {
        src_size = sizeof(fallback);
        src = fallback;
    }
    void *copy = malloc(src_size + 1U);
    if (copy == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    memcpy(copy, src, src_size);
    ((char *)copy)[src_size] = '\0';
    free(state->image);
    state->image = copy;
    state->image_size = src_size;
    state->input_count++;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut)
{
    (void)numOptions;
    (void)options;
    (void)optionValues;
    if (stateOut == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUlinkState state = calloc(1, sizeof(*state));
    if (state == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    state->magic = HANDLE_MAGIC_LINK;
    *stateOut = state;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut)
{
    return cuLinkCreate_v2(numOptions, options, optionValues, stateOut);
}

CUresult CUDAAPI cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name,
                                  unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    (void)type;
    (void)name;
    (void)numOptions;
    (void)options;
    (void)optionValues;
    return link_set_image(state, data, size);
}

CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name,
                               unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    return cuLinkAddData_v2(state, type, data, size, name, numOptions, options, optionValues);
}

CUresult CUDAAPI cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char *path,
                                  unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    (void)type;
    (void)numOptions;
    (void)options;
    (void)optionValues;
    if (!valid_link_state(state) || path == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    size_t size = 0;
    char *data = read_file_bytes(path, &size);
    if (data == NULL) {
        return CUDA_ERROR_FILE_NOT_FOUND;
    }
    CUresult result = link_set_image(state, data, size);
    free(data);
    return result;
}

CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type, const char *path,
                               unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    return cuLinkAddFile_v2(state, type, path, numOptions, options, optionValues);
}

CUresult CUDAAPI cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut)
{
    if (!valid_link_state(state) || cubinOut == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state->image == NULL) {
        CUresult result = link_set_image(state, NULL, 0);
        if (result != CUDA_SUCCESS) {
            return result;
        }
    }
    *cubinOut = state->image;
    if (sizeOut != NULL) {
        *sizeOut = state->image_size;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLinkDestroy(CUlinkState state)
{
    if (!valid_link_state(state)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    free(state->image);
    state->magic = 0;
    free(state);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleUnload(CUmodule hmod)
{
    if (!valid_module(hmod)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    pthread_mutex_lock(&g_lock);
    struct CUmod_st **prev = (struct CUmod_st **)&g.modules;
    while (*prev != NULL) {
        if (*prev == hmod) {
            *prev = hmod->next;
            break;
        }
        prev = &(*prev)->next;
    }
    pthread_mutex_unlock(&g_lock);
    struct CUfunc_st *fn = hmod->functions;
    while (fn != NULL) {
        struct CUfunc_st *next = fn->next;
        free(fn->name);
        free(fn);
        fn = next;
    }
    struct module_global *global = hmod->globals;
    while (global != NULL) {
        struct module_global *next = global->next;
        cuMemFree_v2(global->dptr);
        free(global->name);
        free(global);
        global = next;
    }
    free(hmod->name);
    free(hmod->owned_image);
    hmod->magic = 0;
    free(hmod);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
{
    if (hfunc == NULL || !valid_module(hmod) || name == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (struct CUfunc_st *fn = hmod->functions; fn != NULL; fn = fn->next) {
        if (strcmp(fn->name, name) == 0) {
            *hfunc = fn;
            return CUDA_SUCCESS;
        }
    }
    CUfunction fn = calloc(1, sizeof(*fn));
    if (fn == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    fn->magic = HANDLE_MAGIC_FUNC;
    fn->name = strdup(name);
    fn->module = hmod;
    fn->next = hmod->functions;
    hmod->functions = fn;
    *hfunc = fn;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetFunctionCount(unsigned int *count, CUmodule mod)
{
    if (count == NULL || !valid_module(mod)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsigned int n = 0;
    for (struct CUfunc_st *fn = mod->functions; fn != NULL; fn = fn->next) {
        n++;
    }
    *count = n;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncGetName(const char **name, CUfunction hfunc)
{
    if (name == NULL || !valid_function(hfunc)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *name = hfunc->name;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncGetModule(CUmodule *hmod, CUfunction hfunc)
{
    if (hmod == NULL || !valid_function(hfunc)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *hmod = hfunc->module;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc)
{
    if (pi == NULL || !valid_function(hfunc)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    switch (attrib) {
    case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
        *pi = 1024;
        return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
    case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
    case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
    case CU_FUNC_ATTRIBUTE_NUM_REGS:
        *pi = 0;
        return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_PTX_VERSION:
        *pi = 120;
        return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_BINARY_VERSION:
        *pi = g.cc_major * 10 + g.cc_minor;
        return CUDA_SUCCESS;
    default:
        return CUDA_ERROR_NOT_SUPPORTED;
    }
}

CUresult CUDAAPI cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value)
{
    (void)attrib;
    (void)value;
    return valid_function(hfunc) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
    (void)config;
    return valid_function(hfunc) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult CUDAAPI cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config)
{
    (void)config;
    return valid_function(hfunc) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

static CUresult make_library_from_module(CUlibrary *library, CUmodule module)
{
    if (library == NULL || !valid_module(module)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUlibrary lib = calloc(1, sizeof(*lib));
    if (lib == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    lib->magic = HANDLE_MAGIC_LIBRARY;
    lib->module = module;
    *library = lib;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLibraryLoadData(CUlibrary *library, const void *code,
                                   CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions,
                                   CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions)
{
    (void)jitOptions;
    (void)jitOptionsValues;
    (void)numJitOptions;
    (void)libraryOptions;
    (void)libraryOptionValues;
    (void)numLibraryOptions;
    if (library == NULL || code == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUmodule module = NULL;
    CUresult result = cuModuleLoadData(&module, code);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    result = make_library_from_module(library, module);
    if (result != CUDA_SUCCESS) {
        cuModuleUnload(module);
    }
    return result;
}

CUresult CUDAAPI cuLibraryLoadFromFile(CUlibrary *library, const char *fileName,
                                       CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions,
                                       CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions)
{
    (void)jitOptions;
    (void)jitOptionsValues;
    (void)numJitOptions;
    (void)libraryOptions;
    (void)libraryOptionValues;
    (void)numLibraryOptions;
    if (library == NULL || fileName == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUmodule module = NULL;
    CUresult result = cuModuleLoad(&module, fileName);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    result = make_library_from_module(library, module);
    if (result != CUDA_SUCCESS) {
        cuModuleUnload(module);
    }
    return result;
}

CUresult CUDAAPI cuLibraryUnload(CUlibrary library)
{
    if (!valid_library(library)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    CUkernel kernel = library->kernels;
    while (kernel != NULL) {
        CUkernel next = kernel->next;
        free(kernel->name);
        kernel->magic = 0;
        free(kernel);
        kernel = next;
    }
    CUmodule module = library->module;
    library->magic = 0;
    free(library);
    if (valid_module(module)) {
        return cuModuleUnload(module);
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library, const char *name)
{
    if (pKernel == NULL || !valid_library(library) || name == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (CUkernel kernel = library->kernels; kernel != NULL; kernel = kernel->next) {
        if (strcmp(kernel->name, name) == 0) {
            *pKernel = kernel;
            return CUDA_SUCCESS;
        }
    }
    CUfunction function = NULL;
    CUresult result = cuModuleGetFunction(&function, library->module, name);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    CUkernel kernel = calloc(1, sizeof(*kernel));
    if (kernel == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    kernel->name = strdup(name);
    if (kernel->name == NULL) {
        free(kernel);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    kernel->magic = HANDLE_MAGIC_KERNEL;
    kernel->library = library;
    kernel->function = function;
    kernel->next = library->kernels;
    library->kernels = kernel;
    *pKernel = kernel;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLibraryGetKernelCount(unsigned int *count, CUlibrary lib)
{
    if (count == NULL || !valid_library(lib)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsigned int n = 0;
    for (CUkernel kernel = lib->kernels; kernel != NULL; kernel = kernel->next) {
        n++;
    }
    *count = n;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLibraryEnumerateKernels(CUkernel *kernels, unsigned int numKernels, CUlibrary lib)
{
    if ((kernels == NULL && numKernels != 0) || !valid_library(lib)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsigned int i = 0;
    for (CUkernel kernel = lib->kernels; kernel != NULL && i < numKernels; kernel = kernel->next) {
        kernels[i++] = kernel;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLibraryGetModule(CUmodule *pMod, CUlibrary library)
{
    if (pMod == NULL || !valid_library(library)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pMod = library->module;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel)
{
    if (pFunc == NULL || !valid_kernel(kernel)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pFunc = kernel->function;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuKernelGetLibrary(CUlibrary *pLib, CUkernel kernel)
{
    if (pLib == NULL || !valid_kernel(kernel)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pLib = kernel->library;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name)
{
    if (!valid_library(library)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return cuModuleGetGlobal_v2(dptr, bytes, library->module, name);
}

CUresult CUDAAPI cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name)
{
    return cuLibraryGetGlobal(dptr, bytes, library, name);
}

CUresult CUDAAPI cuLibraryGetUnifiedFunction(void **fptr, CUlibrary library, const char *symbol)
{
    if (fptr == NULL || !valid_library(library) || symbol == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUkernel kernel = NULL;
    CUresult result = cuLibraryGetKernel(&kernel, library, symbol);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    *fptr = (void *)kernel->function;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuKernelGetAttribute(int *pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev)
{
    (void)dev;
    if (pi == NULL || !valid_kernel(kernel)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return cuFuncGetAttribute(pi, attrib, kernel->function);
}

CUresult CUDAAPI cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev)
{
    (void)dev;
    if (!valid_kernel(kernel)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return cuFuncSetAttribute(kernel->function, attrib, val);
}

CUresult CUDAAPI cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev)
{
    (void)config;
    (void)dev;
    return valid_kernel(kernel) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult CUDAAPI cuKernelGetName(const char **name, CUkernel hfunc)
{
    if (name == NULL || !valid_kernel(hfunc)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *name = hfunc->name;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuKernelGetParamInfo(CUkernel kernel, size_t paramIndex, size_t *paramOffset, size_t *paramSize)
{
    if (!valid_kernel(kernel) || paramOffset == NULL || paramSize == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *paramOffset = paramIndex * sizeof(void *);
    *paramSize = sizeof(void *);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchKernel(CUfunction f,
                                unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void **kernelParams, void **extra)
{
    (void)gridDimX;
    (void)gridDimY;
    (void)gridDimZ;
    (void)blockDimX;
    (void)blockDimY;
    (void)blockDimZ;
    (void)sharedMemBytes;
    (void)kernelParams;
    (void)extra;
    if (!valid_function(f) || !valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    bool noop_success = env_enabled("LANXIN_NVIDIA_CUDA_NOOP_KERNEL");
    bool rm_submit_only = env_enabled("LANXIN_NVIDIA_CUDA_RM_SUBMIT");
    bool rm_pb_submit = env_enabled("LANXIN_NVIDIA_CUDA_PB_SUBMIT");
    bool strict_launch = env_enabled("LANXIN_NVIDIA_CUDA_STRICT_LAUNCH");
    bool default_pb_submit = !noop_success && !rm_submit_only && !env_disabled("LANXIN_NVIDIA_CUDA_PB_SUBMIT");
    pthread_mutex_lock(&g_lock);
    CUresult init = ensure_initialized_locked();
    int submit_rc = -1;
    if (init == CUDA_SUCCESS) {
        if (noop_success || rm_submit_only) {
            submit_rc = rm_submit_noop_locked();
        } else if (rm_pb_submit || default_pb_submit) {
            submit_rc = rm_submit_compute_set_object_locked();
            if (submit_rc != 0) {
                submit_rc = rm_submit_noop_locked();
            }
        }
    }
    pthread_mutex_unlock(&g_lock);
    tracef("RM cuLaunchKernel scaffold(%s) submit_rc=%d noop_success=%d submit_only=%d pb_submit=%d default_pb=%d strict=%d",
           f->name != NULL ? f->name : "<unnamed>", submit_rc,
           noop_success ? 1 : 0, rm_submit_only ? 1 : 0, rm_pb_submit ? 1 : 0,
           default_pb_submit ? 1 : 0, strict_launch ? 1 : 0);
    if (submit_rc != 0) {
        if (noop_success) {
            return CUDA_ERROR_UNKNOWN;
        }
        if (strict_launch || rm_submit_only) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        return CUDA_SUCCESS;
    }
    if (rm_submit_only || strict_launch) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **kernelParams, void **extra)
{
    (void)config;
    return cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, NULL, kernelParams, extra);
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize)
{
    (void)func;
    (void)dynamicSMemSize;
    if (numBlocks == NULL || blockSize <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    int blocks = 1536 / blockSize;
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > 32) {
        blocks = 32;
    }
    *numBlocks = blocks;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags)
{
    (void)flags;
    return cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func,
                                                  CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                                  size_t dynamicSMemSize, int blockSizeLimit)
{
    (void)func;
    (void)blockSizeToDynamicSMemSize;
    (void)dynamicSMemSize;
    if (minGridSize == NULL || blockSize == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    int chosen = blockSizeLimit > 0 && blockSizeLimit < 256 ? blockSizeLimit : 256;
    if (chosen < 1) {
        chosen = 1;
    }
    *blockSize = chosen;
    *minGridSize = g.sm_count > 0 ? g.sm_count : DEFAULT_SM_COUNT;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *blockSize, CUfunction func,
                                                           CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                                           size_t dynamicSMemSize, int blockSizeLimit,
                                                           unsigned int flags)
{
    (void)flags;
    return cuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func,
                                            blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit);
}

CUresult CUDAAPI cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, CUfunction func, int numBlocks, int blockSize)
{
    (void)func;
    (void)numBlocks;
    (void)blockSize;
    if (dynamicSmemSize == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *dynamicSmemSize = 48U * 1024U;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGetErrorName(CUresult error, const char **pStr)
{
    if (pStr == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    switch (error) {
    case CUDA_SUCCESS: *pStr = "CUDA_SUCCESS"; break;
    case CUDA_ERROR_INVALID_VALUE: *pStr = "CUDA_ERROR_INVALID_VALUE"; break;
    case CUDA_ERROR_OUT_OF_MEMORY: *pStr = "CUDA_ERROR_OUT_OF_MEMORY"; break;
    case CUDA_ERROR_NOT_INITIALIZED: *pStr = "CUDA_ERROR_NOT_INITIALIZED"; break;
    case CUDA_ERROR_NO_DEVICE: *pStr = "CUDA_ERROR_NO_DEVICE"; break;
    case CUDA_ERROR_INVALID_DEVICE: *pStr = "CUDA_ERROR_INVALID_DEVICE"; break;
    case CUDA_ERROR_INVALID_CONTEXT: *pStr = "CUDA_ERROR_INVALID_CONTEXT"; break;
    case CUDA_ERROR_INVALID_HANDLE: *pStr = "CUDA_ERROR_INVALID_HANDLE"; break;
    case CUDA_ERROR_FILE_NOT_FOUND: *pStr = "CUDA_ERROR_FILE_NOT_FOUND"; break;
    case CUDA_ERROR_INVALID_IMAGE: *pStr = "CUDA_ERROR_INVALID_IMAGE"; break;
    case CUDA_ERROR_INVALID_PTX: *pStr = "CUDA_ERROR_INVALID_PTX"; break;
    case CUDA_ERROR_NO_BINARY_FOR_GPU: *pStr = "CUDA_ERROR_NO_BINARY_FOR_GPU"; break;
    case CUDA_ERROR_JIT_COMPILER_NOT_FOUND: *pStr = "CUDA_ERROR_JIT_COMPILER_NOT_FOUND"; break;
    case CUDA_ERROR_NOT_FOUND: *pStr = "CUDA_ERROR_NOT_FOUND"; break;
    case CUDA_ERROR_NOT_SUPPORTED: *pStr = "CUDA_ERROR_NOT_SUPPORTED"; break;
    default: *pStr = "CUDA_ERROR_UNKNOWN"; break;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGetErrorString(CUresult error, const char **pStr)
{
    if (pStr == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    switch (error) {
    case CUDA_SUCCESS: *pStr = "success"; break;
    case CUDA_ERROR_INVALID_VALUE: *pStr = "invalid value"; break;
    case CUDA_ERROR_OUT_OF_MEMORY: *pStr = "out of memory"; break;
    case CUDA_ERROR_NOT_INITIALIZED: *pStr = "not initialized"; break;
    case CUDA_ERROR_NO_DEVICE: *pStr = "no CUDA-capable device"; break;
    case CUDA_ERROR_INVALID_DEVICE: *pStr = "invalid device"; break;
    case CUDA_ERROR_INVALID_CONTEXT: *pStr = "invalid context"; break;
    case CUDA_ERROR_INVALID_HANDLE: *pStr = "invalid handle"; break;
    case CUDA_ERROR_FILE_NOT_FOUND: *pStr = "file not found"; break;
    case CUDA_ERROR_INVALID_IMAGE: *pStr = "invalid image"; break;
    case CUDA_ERROR_INVALID_PTX: *pStr = "invalid PTX"; break;
    case CUDA_ERROR_NO_BINARY_FOR_GPU: *pStr = "no binary for GPU"; break;
    case CUDA_ERROR_JIT_COMPILER_NOT_FOUND: *pStr = "JIT compiler not found"; break;
    case CUDA_ERROR_NOT_FOUND: *pStr = "not found"; break;
    case CUDA_ERROR_NOT_SUPPORTED: *pStr = "operation not supported by this shim"; break;
    default: *pStr = "unknown CUDA error"; break;
    }
    return CUDA_SUCCESS;
}

struct proc_entry {
    const char *name;
    void *fn;
};

#define PROC_ENTRY(sym) { #sym, (void *)(uintptr_t)&sym }
#define PROC_ALIAS(name, sym) { name, (void *)(uintptr_t)&sym }

static const struct proc_entry proc_table[] = {
    PROC_ENTRY(cuInit),
    PROC_ENTRY(cuDriverGetVersion),
    PROC_ENTRY(cuDeviceGetCount),
    PROC_ENTRY(cuDeviceGet),
    PROC_ENTRY(cuDeviceGetName),
    PROC_ENTRY(cuDeviceGetUuid),
    PROC_ENTRY(cuDeviceGetUuid_v2),
    PROC_ENTRY(cuDeviceTotalMem),
    PROC_ENTRY(cuDeviceTotalMem_v2),
    PROC_ENTRY(cuDeviceGetAttribute),
    PROC_ENTRY(cuDeviceComputeCapability),
    PROC_ENTRY(cuDeviceGetPCIBusId),
    PROC_ENTRY(cuDeviceGetByPCIBusId),
    PROC_ENTRY(cuDevicePrimaryCtxRetain),
    PROC_ENTRY(cuDevicePrimaryCtxRelease),
    PROC_ENTRY(cuDevicePrimaryCtxRelease_v2),
    PROC_ENTRY(cuDevicePrimaryCtxReset),
    PROC_ENTRY(cuDevicePrimaryCtxReset_v2),
    PROC_ENTRY(cuDevicePrimaryCtxSetFlags),
    PROC_ENTRY(cuDevicePrimaryCtxSetFlags_v2),
    PROC_ENTRY(cuDevicePrimaryCtxGetState),
    PROC_ENTRY(cuCtxCreate),
    PROC_ENTRY(cuCtxCreate_v2),
    PROC_ENTRY(cuCtxCreate_v3),
    PROC_ENTRY(cuCtxDestroy),
    PROC_ENTRY(cuCtxDestroy_v2),
    PROC_ENTRY(cuCtxSetCurrent),
    PROC_ENTRY(cuCtxGetCurrent),
    PROC_ENTRY(cuCtxPushCurrent),
    PROC_ENTRY(cuCtxPopCurrent),
    PROC_ENTRY(cuCtxGetDevice),
    PROC_ENTRY(cuCtxGetFlags),
    PROC_ENTRY(cuCtxSynchronize),
    PROC_ENTRY(cuCtxGetApiVersion),
    PROC_ENTRY(cuCtxGetId),
    PROC_ENTRY(cuMemGetInfo),
    PROC_ENTRY(cuMemGetInfo_v2),
    PROC_ENTRY(cuMemAlloc),
    PROC_ENTRY(cuMemAlloc_v2),
    PROC_ENTRY(cuMemAllocManaged),
    PROC_ENTRY(cuMemAllocPitch),
    PROC_ENTRY(cuMemAllocPitch_v2),
    PROC_ENTRY(cuMemFree),
    PROC_ENTRY(cuMemFree_v2),
    PROC_ENTRY(cuMemGetAddressRange),
    PROC_ENTRY(cuMemGetAddressRange_v2),
    PROC_ENTRY(cuMemAllocHost),
    PROC_ENTRY(cuMemAllocHost_v2),
    PROC_ENTRY(cuMemFreeHost),
    PROC_ENTRY(cuMemHostAlloc),
    PROC_ENTRY(cuMemHostRegister),
    PROC_ENTRY(cuMemHostUnregister),
    PROC_ENTRY(cuMemHostGetDevicePointer),
    PROC_ENTRY(cuMemHostGetDevicePointer_v2),
    PROC_ENTRY(cuMemcpy),
    PROC_ENTRY(cuMemcpyAsync),
    PROC_ENTRY(cuMemcpyHtoD),
    PROC_ENTRY(cuMemcpyHtoD_v2),
    PROC_ENTRY(cuMemcpyDtoH),
    PROC_ENTRY(cuMemcpyDtoH_v2),
    PROC_ENTRY(cuMemcpyDtoD),
    PROC_ENTRY(cuMemcpyDtoD_v2),
    PROC_ENTRY(cuMemcpyHtoDAsync),
    PROC_ENTRY(cuMemcpyHtoDAsync_v2),
    PROC_ENTRY(cuMemcpyDtoHAsync),
    PROC_ENTRY(cuMemcpyDtoHAsync_v2),
    PROC_ENTRY(cuMemcpyDtoDAsync),
    PROC_ENTRY(cuMemcpyDtoDAsync_v2),
    PROC_ENTRY(cuMemcpy2D),
    PROC_ENTRY(cuMemcpy2D_v2),
    PROC_ENTRY(cuMemcpy2DUnaligned),
    PROC_ENTRY(cuMemcpy2DUnaligned_v2),
    PROC_ENTRY(cuMemcpy2DAsync),
    PROC_ENTRY(cuMemcpy2DAsync_v2),
    PROC_ENTRY(cuMemsetD8),
    PROC_ENTRY(cuMemsetD8_v2),
    PROC_ENTRY(cuMemsetD16),
    PROC_ENTRY(cuMemsetD16_v2),
    PROC_ENTRY(cuMemsetD32),
    PROC_ENTRY(cuMemsetD32_v2),
    PROC_ENTRY(cuMemsetD8Async),
    PROC_ENTRY(cuMemsetD16Async),
    PROC_ENTRY(cuMemsetD32Async),
    PROC_ENTRY(cuPointerGetAttribute),
    PROC_ENTRY(cuPointerGetAttributes),
    PROC_ENTRY(cuStreamCreate),
    PROC_ENTRY(cuStreamCreateWithPriority),
    PROC_ENTRY(cuStreamDestroy),
    PROC_ENTRY(cuStreamDestroy_v2),
    PROC_ENTRY(cuStreamQuery),
    PROC_ENTRY(cuStreamSynchronize),
    PROC_ENTRY(cuStreamGetPriority),
    PROC_ENTRY(cuStreamGetFlags),
    PROC_ENTRY(cuStreamGetId),
    PROC_ENTRY(cuStreamGetCtx),
    PROC_ENTRY(cuStreamGetDevice),
    PROC_ENTRY(cuEventCreate),
    PROC_ENTRY(cuEventRecord),
    PROC_ENTRY(cuEventRecordWithFlags),
    PROC_ENTRY(cuEventQuery),
    PROC_ENTRY(cuEventSynchronize),
    PROC_ENTRY(cuEventDestroy),
    PROC_ENTRY(cuEventDestroy_v2),
    PROC_ENTRY(cuEventElapsedTime),
    PROC_ENTRY(cuEventElapsedTime_v2),
    PROC_ENTRY(cuModuleLoad),
    PROC_ENTRY(cuModuleLoadData),
    PROC_ENTRY(cuModuleLoadDataEx),
    PROC_ENTRY(cuModuleLoadFatBinary),
    PROC_ENTRY(cuModuleGetGlobal),
    PROC_ENTRY(cuModuleGetGlobal_v2),
    PROC_ENTRY(cuModuleUnload),
    PROC_ENTRY(cuModuleGetFunction),
    PROC_ENTRY(cuModuleGetFunctionCount),
    PROC_ENTRY(cuLinkCreate),
    PROC_ENTRY(cuLinkCreate_v2),
    PROC_ENTRY(cuLinkAddData),
    PROC_ENTRY(cuLinkAddData_v2),
    PROC_ENTRY(cuLinkAddFile),
    PROC_ENTRY(cuLinkAddFile_v2),
    PROC_ENTRY(cuLinkComplete),
    PROC_ENTRY(cuLinkDestroy),
    PROC_ENTRY(cuFuncGetName),
    PROC_ENTRY(cuFuncGetModule),
    PROC_ENTRY(cuFuncGetAttribute),
    PROC_ENTRY(cuFuncSetAttribute),
    PROC_ENTRY(cuFuncSetCacheConfig),
    PROC_ENTRY(cuFuncSetSharedMemConfig),
    PROC_ENTRY(cuLibraryLoadData),
    PROC_ENTRY(cuLibraryLoadFromFile),
    PROC_ENTRY(cuLibraryUnload),
    PROC_ENTRY(cuLibraryGetKernel),
    PROC_ENTRY(cuLibraryGetKernelCount),
    PROC_ENTRY(cuLibraryEnumerateKernels),
    PROC_ENTRY(cuLibraryGetModule),
    PROC_ENTRY(cuKernelGetFunction),
    PROC_ENTRY(cuKernelGetLibrary),
    PROC_ENTRY(cuLibraryGetGlobal),
    PROC_ENTRY(cuLibraryGetManaged),
    PROC_ENTRY(cuLibraryGetUnifiedFunction),
    PROC_ENTRY(cuKernelGetAttribute),
    PROC_ENTRY(cuKernelSetAttribute),
    PROC_ENTRY(cuKernelSetCacheConfig),
    PROC_ENTRY(cuKernelGetName),
    PROC_ENTRY(cuKernelGetParamInfo),
    PROC_ENTRY(cuLaunchKernel),
    PROC_ENTRY(cuLaunchKernelEx),
    PROC_ENTRY(cuOccupancyMaxActiveBlocksPerMultiprocessor),
    PROC_ENTRY(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags),
    PROC_ENTRY(cuOccupancyMaxPotentialBlockSize),
    PROC_ENTRY(cuOccupancyMaxPotentialBlockSizeWithFlags),
    PROC_ENTRY(cuOccupancyAvailableDynamicSMemPerBlock),
    PROC_ENTRY(cuGetErrorName),
    PROC_ENTRY(cuGetErrorString),
};

static CUresult lookup_proc(const char *symbol, void **pfn, CUdriverProcAddressQueryResult *symbolStatus)
{
    if (symbol == NULL || pfn == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (size_t i = 0; i < sizeof(proc_table) / sizeof(proc_table[0]); i++) {
        if (strcmp(symbol, proc_table[i].name) == 0) {
            *pfn = proc_table[i].fn;
            if (symbolStatus != NULL) {
                *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
            }
            return CUDA_SUCCESS;
        }
    }
    if (symbolStatus != NULL) {
        *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    }
    *pfn = NULL;
    return CUDA_ERROR_NOT_FOUND;
}

CUresult CUDAAPI cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus)
{
    (void)cudaVersion;
    (void)flags;
    return lookup_proc(symbol, pfn, symbolStatus);
}

CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags)
{
    return cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, NULL);
}

size_t lanxin_nvidia_cuda_tracked_allocated_bytes(void)
{
    pthread_mutex_lock(&g_lock);
    size_t value = g.allocated_bytes;
    pthread_mutex_unlock(&g_lock);
    return value;
}
