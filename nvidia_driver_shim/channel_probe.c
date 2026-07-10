#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define NV_IOCTL_MAGIC 'F'
#define NV_IOCTL_BASE 200
#define NV_ESC_CARD_INFO (NV_IOCTL_BASE + 0)
#define NV_ESC_REGISTER_FD (NV_IOCTL_BASE + 1)
#define NV_ESC_CHECK_VERSION_STR (NV_IOCTL_BASE + 10)
#define NV_ESC_IOCTL_XFER_CMD (NV_IOCTL_BASE + 11)
#define NV_ESC_ATTACH_GPUS_TO_FD (NV_IOCTL_BASE + 12)
#define NV_ESC_WAIT_OPEN_COMPLETE (NV_IOCTL_BASE + 18)

#define NV_ESC_RM_ALLOC_MEMORY 0x27
#define NV_ESC_RM_FREE 0x29
#define NV_ESC_RM_CONTROL 0x2a
#define NV_ESC_RM_ALLOC 0x2b
#define NV_ESC_RM_MAP_MEMORY 0x4e
#define NV_ESC_RM_UNMAP_MEMORY 0x4f
#define NV_ESC_RM_ALLOC_CONTEXT_DMA2 0x54
#define NV_ESC_RM_MAP_MEMORY_DMA 0x57
#define NV_ESC_RM_UNMAP_MEMORY_DMA 0x58

#define NV01_NULL_OBJECT 0x0u
#define NV01_ROOT 0x0u
#define NV01_DEVICE_0 0x80u
#define NV20_SUBDEVICE_0 0x2080u
#define NV01_CONTEXT_DMA 0x2u
#define NV01_MEMORY_SYSTEM 0x3eu
#define NV01_MEMORY_LOCAL_USER 0x40u
#define NV01_MEMORY_VIRTUAL 0x70u

#define BLACKWELL_CHANNEL_GPFIFO_B 0xca6fu
#define BLACKWELL_CHANNEL_GPFIFO_A 0xc96fu
#define HOPPER_CHANNEL_GPFIFO_A 0xc86fu
#define AMPERE_CHANNEL_GPFIFO_A 0xc56fu
#define BLACKWELL_COMPUTE_B 0xcec0u
#define BLACKWELL_COMPUTE_A 0xcdc0u
#define HOPPER_COMPUTE_A 0xcbc0u
#define AMPERE_COMPUTE_B 0xc7c0u
#define AMPERE_COMPUTE_A 0xc6c0u

#define NV_OK 0u
#define NV_MAX_SUBDEVICES 8u
#define NV2080_ENGINE_TYPE_GRAPHICS 0x1u

#define NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS (1u << 4)
#define NVOS02_FLAGS_LOCATION_PCI (0u << 8)
#define NVOS02_FLAGS_LOCATION_VIDMEM (2u << 8)
#define NVOS02_FLAGS_COHERENCY_WRITE_COMBINE (2u << 12)
#define NVOS02_FLAGS_COHERENCY_WRITE_BACK (5u << 12)
#define NVOS02_FLAGS_GPU_CACHEABLE_YES (1u << 18)
#define NVOS02_FLAGS_MAPPING_NO_MAP (1u << 30)

#define NVOS03_FLAGS_MAPPING_KERNEL (1u << 20)
#define NVOS03_FLAGS_HASH_TABLE_DISABLE (1u << 29)

#define NVOS46_FLAGS_CACHE_SNOOP_ENABLE (1u << 4)
#define NVOS46_FLAGS_PAGE_SIZE_4KB (1u << 8)
#define NVOS04_FLAGS_MAP_CHANNEL_TRUE (1u << 30)

#define NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES 0u
#define NVA06F_CTRL_CMD_GPFIFO_SCHEDULE 0xa06f0103u
#define NVA06F_CTRL_CMD_BIND 0xa06f0104u
#define NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN 0xc36f0108u
#define NV906F_CTRL_GET_CLASS_ENGINEID 0x906f0101u

#define NVA06F_SUBCHANNEL_COMPUTE 1u
#define NVC46F_DMA_SEC_OP_INC_METHOD 1u
#define NV_COMPUTE_SET_OBJECT 0x0000u
#define NV_COMPUTE_NO_OPERATION 0x0100u
#define NV_COMPUTE_PIPE_NOP 0x1a2cu

typedef uint8_t NvBool;
typedef uint32_t NvU32;
typedef int32_t NvV32;
typedef uint64_t NvU64;
typedef uint64_t NvP64;
typedef uint32_t NvHandle;

typedef struct {
    NvU32 domain;
    uint8_t bus;
    uint8_t slot;
    uint8_t function;
    uint16_t vendor_id;
    uint16_t device_id;
} nv_pci_info_t;

typedef struct {
    NvU32 cmd;
    NvU32 size;
    NvP64 ptr __attribute__((aligned(8)));
} nv_ioctl_xfer_t;

typedef struct {
    uint8_t valid;
    nv_pci_info_t pci_info;
    NvU32 gpu_id;
    uint16_t interrupt_line;
    NvU64 reg_address __attribute__((aligned(8)));
    NvU64 reg_size __attribute__((aligned(8)));
    NvU64 fb_address __attribute__((aligned(8)));
    NvU64 fb_size __attribute__((aligned(8)));
    NvU32 minor_number;
    uint8_t dev_name[10];
} nv_ioctl_card_info_t;

typedef struct {
    NvU32 cmd;
    NvU32 reply;
    char versionString[64];
} nv_ioctl_rm_api_version_t;

typedef struct {
    int rc;
    NvU32 adapterStatus;
} nv_ioctl_wait_open_complete_t;

typedef struct {
    int ctl_fd;
} nv_ioctl_register_fd_t;

typedef struct {
    NvHandle hRoot;
    NvHandle hObjectParent;
    NvHandle hObjectOld;
    NvV32 status;
} NVOS00_PARAMETERS;

typedef struct {
    NvHandle hRoot;
    NvHandle hObjectParent;
    NvHandle hObjectNew;
    NvV32 hClass;
    NvV32 flags;
    NvP64 pMemory __attribute__((aligned(8)));
    NvU64 limit __attribute__((aligned(8)));
    NvV32 status;
} NVOS02_PARAMETERS;

typedef struct {
    NVOS02_PARAMETERS params;
    int fd;
} nv_ioctl_nvos02_parameters_with_fd;

typedef struct {
    NvHandle hRoot;
    NvHandle hObjectParent;
    NvHandle hObjectNew;
    NvV32 hClass;
    NvP64 pAllocParms __attribute__((aligned(8)));
    NvU32 paramsSize;
    NvV32 status;
} NVOS21_PARAMETERS;

typedef struct {
    NvHandle hClient;
    NvHandle hDevice;
    NvHandle hMemory;
    NvU64 offset __attribute__((aligned(8)));
    NvU64 length __attribute__((aligned(8)));
    NvP64 pLinearAddress __attribute__((aligned(8)));
    NvU32 status;
    NvU32 flags;
} NVOS33_PARAMETERS;

typedef struct {
    NVOS33_PARAMETERS params;
    int fd;
} nv_ioctl_nvos33_parameters_with_fd;

typedef struct {
    NvHandle hClient;
    NvHandle hDevice;
    NvHandle hMemory;
    NvP64 pLinearAddress __attribute__((aligned(8)));
    NvU32 status;
    NvU32 flags;
} NVOS34_PARAMETERS;

typedef struct {
    NvHandle hObjectParent;
    NvHandle hSubDevice;
    NvHandle hObjectNew;
    NvV32 hClass;
    NvV32 flags;
    NvU32 selector;
    NvHandle hMemory;
    NvU64 offset __attribute__((aligned(8)));
    NvU64 limit __attribute__((aligned(8)));
    NvV32 status;
} NVOS39_PARAMETERS;

typedef struct {
    NvHandle hClient;
    NvHandle hObject;
    NvV32 cmd;
    NvU32 flags;
    NvP64 params __attribute__((aligned(8)));
    NvU32 paramsSize;
    NvV32 status;
} NVOS54_PARAMETERS;

typedef struct {
    NvHandle hClient;
    NvHandle hDevice;
    NvHandle hDma;
    NvHandle hMemory;
    NvU64 offset __attribute__((aligned(8)));
    NvU64 length __attribute__((aligned(8)));
    NvV32 flags;
    NvU64 dmaOffset __attribute__((aligned(8)));
    NvV32 status;
} NVOS46_PARAMETERS;

typedef struct {
    NvHandle hClient;
    NvHandle hDevice;
    NvHandle hDma;
    NvHandle hMemory;
    NvV32 flags;
    NvU64 dmaOffset __attribute__((aligned(8)));
    NvU64 size __attribute__((aligned(8)));
    NvV32 status;
} NVOS47_PARAMETERS;

typedef struct {
    NvU32 deviceId;
    NvHandle hClientShare;
    NvHandle hTargetClient;
    NvHandle hTargetDevice;
    NvV32 flags;
    NvU64 vaSpaceSize __attribute__((aligned(8)));
    NvU64 vaStartInternal __attribute__((aligned(8)));
    NvU64 vaLimitInternal __attribute__((aligned(8)));
    NvV32 vaMode;
} NV0080_ALLOC_PARAMETERS;

typedef struct {
    NvU32 subDeviceId;
} NV2080_ALLOC_PARAMETERS;

typedef struct {
    NvU64 offset __attribute__((aligned(8)));
    NvU64 limit __attribute__((aligned(8)));
    NvHandle hVASpace;
} NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS;

typedef struct {
    NvU64 base __attribute__((aligned(8)));
    NvU64 size __attribute__((aligned(8)));
    NvU32 addressSpace;
    NvU32 cacheAttrib;
} NV_MEMORY_DESC_PARAMS;

typedef struct {
    NvHandle hObjectError;
    NvHandle hObjectBuffer;
    NvU64 gpFifoOffset __attribute__((aligned(8)));
    NvU32 gpFifoEntries;
    NvU32 flags;
    NvHandle hContextShare;
    NvHandle hVASpace;
    NvHandle hUserdMemory[NV_MAX_SUBDEVICES];
    NvU64 userdOffset[NV_MAX_SUBDEVICES] __attribute__((aligned(8)));
    NvU32 engineType;
    NvU32 cid;
    NvU32 subDeviceId;
    NvHandle hObjectEccError;
    NV_MEMORY_DESC_PARAMS instanceMem __attribute__((aligned(8)));
    NV_MEMORY_DESC_PARAMS userdMem __attribute__((aligned(8)));
    NV_MEMORY_DESC_PARAMS ramfcMem __attribute__((aligned(8)));
    NV_MEMORY_DESC_PARAMS mthdbufMem __attribute__((aligned(8)));
    NvHandle hPhysChannelGroup;
    NvU32 internalFlags;
    NV_MEMORY_DESC_PARAMS errorNotifierMem __attribute__((aligned(8)));
    NV_MEMORY_DESC_PARAMS eccErrorNotifierMem __attribute__((aligned(8)));
    NvU32 ProcessID;
    NvU32 SubProcessID;
    NvU32 encryptIv[3];
    NvU32 decryptIv[3];
    NvU32 hmacNonce[8];
} NV_CHANNEL_ALLOC_PARAMS;

typedef struct {
    NvU32 engineType;
} NVA06F_CTRL_BIND_PARAMS;

typedef struct {
    NvBool bEnable;
    NvBool bSkipSubmit;
    NvBool bSkipEnable;
} NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS;

typedef struct {
    NvU32 workSubmitToken;
} NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS;

typedef struct {
    NvHandle hObject;
    NvU32 classEngineID;
    NvU32 classID;
    NvU32 engineID;
} NV906F_CTRL_GET_CLASS_ENGINEID_PARAMS;

static unsigned long nv_iowr(unsigned int nr, size_t size)
{
    return _IOC(_IOC_READ | _IOC_WRITE, NV_IOCTL_MAGIC, nr, size);
}

static int rm_ioctl_xfer(int fd, NvU32 cmd, void *data, size_t size)
{
    nv_ioctl_xfer_t xfer = {
        .cmd = cmd,
        .size = (NvU32)size,
        .ptr = (NvP64)(uintptr_t)data,
    };
    return ioctl(fd, nv_iowr(NV_ESC_IOCTL_XFER_CMD, sizeof(xfer)), &xfer);
}

static int rm_ioctl_direct(int fd, NvU32 cmd, void *data, size_t size)
{
    return ioctl(fd, nv_iowr(cmd, size), data);
}

static void print_status(const char *name, int rc, NvU32 status)
{
    if (rc == 0) {
        printf("%s ioctl=ok rm_status=0x%08x\n", name, status);
    } else {
        printf("%s ioctl=err errno=%d:%s rm_status=0x%08x\n", name, errno, strerror(errno), status);
    }
}

static int rm_alloc(int ctl_fd, NvHandle hRoot, NvHandle hParent, NvHandle *hObject,
                    NvU32 hClass, void *params, NvU32 paramsSize, const char *label)
{
    NVOS21_PARAMETERS api;
    memset(&api, 0, sizeof(api));
    api.hRoot = hRoot;
    api.hObjectParent = hParent;
    api.hObjectNew = hObject != NULL ? *hObject : 0;
    api.hClass = (NvV32)hClass;
    api.pAllocParms = (NvP64)(uintptr_t)params;
    api.paramsSize = paramsSize;
    int rc = rm_ioctl_xfer(ctl_fd, NV_ESC_RM_ALLOC, &api, sizeof(api));
    print_status(label, rc, (NvU32)api.status);
    printf("  hRoot=0x%x hParent=0x%x hObjectNew=0x%x class=0x%x paramsSize=%u\n",
           api.hRoot, api.hObjectParent, api.hObjectNew, (NvU32)api.hClass, api.paramsSize);
    if (rc == 0 && api.status == NV_OK && hObject != NULL) {
        *hObject = api.hObjectNew;
    }
    return rc == 0 && api.status == NV_OK ? 0 : -1;
}

static int rm_free(int ctl_fd, NvHandle hRoot, NvHandle hParent, NvHandle hObject, const char *label)
{
    if (hObject == 0) {
        return 0;
    }
    NVOS00_PARAMETERS api;
    memset(&api, 0, sizeof(api));
    api.hRoot = hRoot;
    api.hObjectParent = hParent;
    api.hObjectOld = hObject;
    int rc = rm_ioctl_xfer(ctl_fd, NV_ESC_RM_FREE, &api, sizeof(api));
    print_status(label, rc, (NvU32)api.status);
    return rc == 0 && api.status == NV_OK ? 0 : -1;
}

static int rm_alloc_memory(int ioctl_fd, int map_fd, NvHandle client, NvHandle parent, NvHandle *hMemory,
                           NvU32 hClass, NvU32 flags, size_t size, const char *label)
{
    nv_ioctl_nvos02_parameters_with_fd api;
    memset(&api, 0, sizeof(api));
    api.params.hRoot = client;
    api.params.hObjectParent = parent;
    api.params.hObjectNew = *hMemory;
    api.params.hClass = (NvV32)hClass;
    api.params.flags = (NvV32)flags;
    api.params.pMemory = 0;
    api.params.limit = (NvU64)size - 1;
    api.fd = map_fd;
    int rc = rm_ioctl_xfer(ioctl_fd, NV_ESC_RM_ALLOC_MEMORY, &api, sizeof(api));
    print_status(label, rc, (NvU32)api.params.status);
    printf("  hMemory=0x%x class=0x%x flags=0x%08x pMemory=0x%016" PRIx64 " limit=0x%016" PRIx64 " fd=%d\n",
           api.params.hObjectNew, (NvU32)api.params.hClass, (NvU32)api.params.flags,
           api.params.pMemory, api.params.limit, api.fd);
    if (rc == 0 && api.params.status == NV_OK) {
        *hMemory = api.params.hObjectNew;
        return 0;
    }
    return -1;
}

static int rm_map_memory(int ctl_fd, int map_fd, NvHandle client, NvHandle device, NvHandle hMemory,
                         size_t size, NvP64 *linear, const char *label)
{
    nv_ioctl_nvos33_parameters_with_fd api;
    memset(&api, 0, sizeof(api));
    api.params.hClient = client;
    api.params.hDevice = device;
    api.params.hMemory = hMemory;
    api.params.offset = 0;
    api.params.length = size;
    api.params.pLinearAddress = 0;
    api.params.flags = 0;
    api.fd = map_fd;
    int rc = rm_ioctl_xfer(ctl_fd, NV_ESC_RM_MAP_MEMORY, &api, sizeof(api));
    print_status(label, rc, api.params.status);
    printf("  hDevice=0x%x hMemory=0x%x pLinearAddress=0x%016" PRIx64 " length=0x%zx fd=%d\n",
           device, hMemory, api.params.pLinearAddress, size, api.fd);
    if (rc == 0 && api.params.status == NV_OK) {
        *linear = api.params.pLinearAddress;
        return 0;
    }
    return -1;
}

static int rm_unmap_memory(int ctl_fd, NvHandle client, NvHandle device, NvHandle hMemory,
                           NvP64 linear, const char *label)
{
    if (linear == 0 || hMemory == 0) {
        return 0;
    }
    NVOS34_PARAMETERS api;
    memset(&api, 0, sizeof(api));
    api.hClient = client;
    api.hDevice = device;
    api.hMemory = hMemory;
    api.pLinearAddress = linear;
    int rc = rm_ioctl_xfer(ctl_fd, NV_ESC_RM_UNMAP_MEMORY, &api, sizeof(api));
    print_status(label, rc, api.status);
    return rc == 0 && api.status == NV_OK ? 0 : -1;
}

static int rm_map_memory_dma(int ctl_fd, NvHandle client, NvHandle device, NvHandle hDma,
                             NvHandle hMemory, size_t size, NvU32 flags, NvU64 *gpu_va,
                             const char *label)
{
    NVOS46_PARAMETERS api;
    memset(&api, 0, sizeof(api));
    api.hClient = client;
    api.hDevice = device;
    api.hDma = hDma;
    api.hMemory = hMemory;
    api.offset = 0;
    api.length = size;
    api.flags = (NvV32)flags;
    api.dmaOffset = gpu_va != NULL ? *gpu_va : 0;
    int rc = rm_ioctl_xfer(ctl_fd, NV_ESC_RM_MAP_MEMORY_DMA, &api, sizeof(api));
    print_status(label, rc, (NvU32)api.status);
    printf("  hDevice=0x%x hDma=0x%x hMemory=0x%x size=0x%zx flags=0x%08x dmaOffset=0x%016" PRIx64 "\n",
           device, hDma, hMemory, size, flags, api.dmaOffset);
    if (rc == 0 && api.status == NV_OK) {
        if (gpu_va != NULL) {
            *gpu_va = api.dmaOffset;
        }
        return 0;
    }
    return -1;
}

static int rm_unmap_memory_dma(int ctl_fd, NvHandle client, NvHandle device, NvHandle hDma,
                               NvHandle hMemory, NvU64 gpu_va, size_t size, const char *label)
{
    if (gpu_va == 0 || hDma == 0 || hMemory == 0) {
        return 0;
    }
    NVOS47_PARAMETERS api;
    memset(&api, 0, sizeof(api));
    api.hClient = client;
    api.hDevice = device;
    api.hDma = hDma;
    api.hMemory = hMemory;
    api.dmaOffset = gpu_va;
    api.size = size;
    int rc = rm_ioctl_xfer(ctl_fd, NV_ESC_RM_UNMAP_MEMORY_DMA, &api, sizeof(api));
    print_status(label, rc, (NvU32)api.status);
    return rc == 0 && api.status == NV_OK ? 0 : -1;
}

static int rm_control(int ctl_fd, NvHandle client, NvHandle object, NvU32 cmd,
                      void *params, NvU32 paramsSize, const char *label)
{
    NVOS54_PARAMETERS api;
    memset(&api, 0, sizeof(api));
    api.hClient = client;
    api.hObject = object;
    api.cmd = (NvV32)cmd;
    api.flags = 0;
    api.params = (NvP64)(uintptr_t)params;
    api.paramsSize = paramsSize;
    int rc = rm_ioctl_xfer(ctl_fd, NV_ESC_RM_CONTROL, &api, sizeof(api));
    print_status(label, rc, (NvU32)api.status);
    printf("  hObject=0x%x cmd=0x%08x paramsSize=%u\n", object, cmd, paramsSize);
    return rc == 0 && api.status == NV_OK ? 0 : -1;
}

static void *map_cpu_from_fd(int fd, size_t size, const char *label)
{
    void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    printf("%s mmap fd=%d -> %p errno=%d:%s\n",
           label, fd, addr, addr == MAP_FAILED ? errno : 0,
           addr == MAP_FAILED ? strerror(errno) : "ok");
    return addr == MAP_FAILED ? NULL : addr;
}

static int try_map_dma_two_devices(int ctl_fd, NvHandle client, NvHandle device, NvHandle subdevice,
                                   NvHandle hDma, NvHandle hMemory, size_t size, NvU32 flags,
                                   NvU64 *gpu_va, const char *label)
{
    char buf[128];
    snprintf(buf, sizeof(buf), "%s_device", label);
    if (rm_map_memory_dma(ctl_fd, client, device, hDma, hMemory, size, flags, gpu_va, buf) == 0) {
        return 0;
    }
    if (subdevice != 0) {
        snprintf(buf, sizeof(buf), "%s_subdevice", label);
        if (rm_map_memory_dma(ctl_fd, client, subdevice, hDma, hMemory, size, flags, gpu_va, buf) == 0) {
            return 0;
        }
    }
    return -1;
}

static NvHandle alloc_client(int ctl_fd)
{
    NvHandle client = 0;
    NvU32 root_out = 0;
    if (rm_alloc(ctl_fd, NV01_NULL_OBJECT, NV01_NULL_OBJECT, &client, NV01_ROOT,
                 &root_out, sizeof(root_out), "alloc_root:size4") != 0) {
        client = 0;
        root_out = 0;
        (void)rm_alloc(ctl_fd, NV01_NULL_OBJECT, NV01_NULL_OBJECT, &client, NV01_ROOT,
                       &root_out, 0, "alloc_root:size0");
    }
    printf("root_out=0x%x client=0x%x\n", root_out, client);
    if (client == 0 && root_out != 0) {
        client = root_out;
    }
    return client;
}

static int setup_device(int ctl_fd, NvHandle client, NvHandle *device, NvHandle *subdevice)
{
    *device = 0x1000;
    NV0080_ALLOC_PARAMETERS dev_params;
    memset(&dev_params, 0, sizeof(dev_params));
    dev_params.deviceId = 0;
    dev_params.hClientShare = client;
    dev_params.vaMode = NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES;
    if (rm_alloc(ctl_fd, client, client, device, NV01_DEVICE_0, &dev_params,
                 sizeof(dev_params), "alloc_device") != 0) {
        return -1;
    }

    *subdevice = 0x2000;
    NV2080_ALLOC_PARAMETERS sub_params = {.subDeviceId = 0};
    if (rm_alloc(ctl_fd, client, *device, subdevice, NV20_SUBDEVICE_0, &sub_params,
                 sizeof(sub_params), "alloc_subdevice") != 0) {
        *subdevice = 0;
    }
    return 0;
}

static int allocate_vaspace(int ctl_fd, NvHandle client, NvHandle device, NvHandle *vaspace)
{
    NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS params;
    memset(&params, 0, sizeof(params));
    params.offset = 0;
    params.limit = 0;
    params.hVASpace = 0;
    *vaspace = 0x5000;
    if (rm_alloc(ctl_fd, client, device, vaspace, NV01_MEMORY_VIRTUAL, &params,
                 sizeof(params), "alloc_nv01_memory_virtual") == 0) {
        printf("  vaspace=0x%x returned_limit=0x%016" PRIx64 "\n", *vaspace, params.limit);
        return 0;
    }
    *vaspace = 0;
    return -1;
}

static int allocate_ctxdma(int ctl_fd, NvHandle client, NvHandle device, NvHandle subdevice,
                           NvHandle hMemory, NvU64 limit, NvU32 flags, NvHandle *ctxdma,
                           const char *label)
{
    const struct {
        NvHandle parent;
        NvHandle subdev;
        const char *suffix;
    } tries[] = {
        {device, subdevice, "dev_subdev"},
        {client, subdevice, "client_subdev"},
        {device, device, "dev_device"},
        {client, device, "client_device"},
        {device, 0, "dev_null"},
        {client, 0, "client_null"},
    };

    for (size_t i = 0; i < sizeof(tries) / sizeof(tries[0]); i++) {
        NVOS39_PARAMETERS api;
        char full_label[128];
        memset(&api, 0, sizeof(api));
        api.hObjectParent = tries[i].parent;
        api.hSubDevice = tries[i].subdev;
        api.hObjectNew = *ctxdma;
        api.hClass = NV01_CONTEXT_DMA;
        api.flags = (NvV32)flags;
        api.selector = 0;
        api.hMemory = hMemory;
        api.offset = 0;
        api.limit = limit;
        snprintf(full_label, sizeof(full_label), "%s_%s", label, tries[i].suffix);
        int rc = rm_ioctl_xfer(ctl_fd, NV_ESC_RM_ALLOC_CONTEXT_DMA2, &api, sizeof(api));
        print_status(full_label, rc, (NvU32)api.status);
        printf("  parent=0x%x subdevice=0x%x ctxdma=0x%x class=0x%x hMemory=0x%x limit=0x%016" PRIx64 " flags=0x%08x\n",
               api.hObjectParent, api.hSubDevice, api.hObjectNew, (NvU32)api.hClass,
               api.hMemory, api.limit, (NvU32)api.flags);
        if (rc == 0 && api.status == NV_OK) {
            *ctxdma = api.hObjectNew;
            return 0;
        }
    }
    return -1;
}

static int try_channel_alloc(int ctl_fd, NvHandle client, NvHandle device, NvHandle hDma,
                             NvHandle vaspace, NvHandle error_ctxdma, NvHandle userd,
                             NvU64 gp_fifo_va, NvHandle *channel, NvU32 *class_used)
{
    static const NvU32 classes[] = {
        BLACKWELL_CHANNEL_GPFIFO_B,
        BLACKWELL_CHANNEL_GPFIFO_A,
        HOPPER_CHANNEL_GPFIFO_A,
        AMPERE_CHANNEL_GPFIFO_A,
    };
    const struct {
        int use_error;
        int use_legacy_buffer;
        int use_vaspace;
        const char *suffix;
    } modes[] = {
        {1, 1, 0, "legacy"},
        {1, 0, 1, "hvaspace"},
        {0, 1, 0, "legacy_noerr"},
        {0, 0, 1, "hvaspace_noerr"},
    };

    for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++) {
        if (modes[m].use_legacy_buffer && hDma == 0) {
            continue;
        }
        if (modes[m].use_vaspace && vaspace == 0) {
            continue;
        }
        if (modes[m].use_error && error_ctxdma == 0) {
            continue;
        }
        for (size_t i = 0; i < sizeof(classes) / sizeof(classes[0]); i++) {
        NV_CHANNEL_ALLOC_PARAMS params;
        memset(&params, 0, sizeof(params));
        params.hObjectError = modes[m].use_error ? error_ctxdma : 0;
        if (modes[m].use_vaspace) {
            params.hVASpace = vaspace;
        }
        if (modes[m].use_legacy_buffer) {
            params.hObjectBuffer = hDma;
        }
        params.gpFifoOffset = gp_fifo_va;
        params.gpFifoEntries = 32;
        params.flags = NVOS04_FLAGS_MAP_CHANNEL_TRUE;
        params.engineType = NV2080_ENGINE_TYPE_GRAPHICS;
        params.ProcessID = (NvU32)getpid();
        if (userd != 0) {
            params.hUserdMemory[0] = userd;
            params.userdOffset[0] = 0;
        }

        *channel = 0x7000 + (NvHandle)(m * 0x10 + i);
        char label[96];
        snprintf(label, sizeof(label), "alloc_channel_%s_class_0x%04x%s",
                 modes[m].suffix, classes[i], userd != 0 ? "_userd" : "");
        if (rm_alloc(ctl_fd, client, device, channel, classes[i], &params,
                     sizeof(params), label) == 0) {
            *class_used = classes[i];
            printf("  channel=0x%x class=0x%04x gpFifoVA=0x%016" PRIx64
                   " hDma=0x%x hVASpace=0x%x errorCtx=0x%x userd=0x%x cid=%u\n",
                   *channel, *class_used, gp_fifo_va, hDma, vaspace, error_ctxdma,
                   userd, params.cid);
            return 0;
        }
        }
    }
    *channel = 0;
    return -1;
}

static NvU32 push_method_header(NvU32 subchannel, NvU32 method, NvU32 count, NvU32 sec_op)
{
    return ((sec_op & 0x7u) << 29) |
           ((count & 0x1fffu) << 16) |
           ((subchannel & 0x7u) << 13) |
           ((method >> 2) & 0xfffu);
}

static void write_gpfifo_entry(volatile NvU32 *gp_words, NvU32 entry,
                               NvU64 pushbuffer_va, NvU32 pushbuffer_bytes)
{
    gp_words[entry * 2U] = (NvU32)(pushbuffer_va & 0xfffffffcu);
    gp_words[entry * 2U + 1U] = (NvU32)(((pushbuffer_va >> 32) & 0xffu) |
                                         (((NvU64)(pushbuffer_bytes / 4U) & 0x1fffffu) << 10));
}

static int try_compute_alloc(int ctl_fd, NvHandle client, NvHandle channel,
                             NvHandle *compute, NvU32 *class_used,
                             NvU32 *class_engine_id)
{
    static const NvU32 classes[] = {
        BLACKWELL_COMPUTE_B,
        BLACKWELL_COMPUTE_A,
        HOPPER_COMPUTE_A,
        AMPERE_COMPUTE_B,
        AMPERE_COMPUTE_A,
    };

    for (size_t i = 0; i < sizeof(classes) / sizeof(classes[0]); i++) {
        *compute = 0x7200 + (NvHandle)i;
        char label[96];
        snprintf(label, sizeof(label), "alloc_compute_class_0x%04x", classes[i]);
        if (rm_alloc(ctl_fd, client, channel, compute, classes[i], NULL, 0, label) != 0) {
            continue;
        }

        NV906F_CTRL_GET_CLASS_ENGINEID_PARAMS params;
        memset(&params, 0, sizeof(params));
        params.hObject = *compute;
        if (rm_control(ctl_fd, client, channel, NV906F_CTRL_GET_CLASS_ENGINEID,
                       &params, sizeof(params), "control_get_compute_class_engine_id") != 0 ||
            params.classEngineID == 0) {
            printf("  compute=0x%x class=0x%04x unusable classEngineID=0x%08x classID=0x%08x engineID=0x%08x\n",
                   *compute, classes[i], params.classEngineID, params.classID, params.engineID);
            (void)rm_free(ctl_fd, client, channel, *compute, "free_compute_no_engine_id");
            *compute = 0;
            continue;
        }

        *class_used = classes[i];
        *class_engine_id = params.classEngineID;
        printf("  compute=0x%x class=0x%04x classEngineID=0x%08x classID=0x%08x engineID=0x%08x\n",
               *compute, *class_used, params.classEngineID, params.classID, params.engineID);
        return 0;
    }

    *compute = 0;
    *class_used = 0;
    *class_engine_id = 0;
    return -1;
}

static void submit_compute_set_object_pb(volatile NvU32 *gp_words, volatile NvU32 *control,
                                         NvU64 gp_va, NvU32 *put,
                                         NvU32 class_engine_id)
{
    const NvU32 pb_offset = 0x200u;
    volatile NvU32 *pb_words = (volatile NvU32 *)((volatile uint8_t *)gp_words + pb_offset);

    pb_words[0] = push_method_header(NVA06F_SUBCHANNEL_COMPUTE, NV_COMPUTE_SET_OBJECT,
                                     1, NVC46F_DMA_SEC_OP_INC_METHOD);
    pb_words[1] = class_engine_id;
    pb_words[2] = push_method_header(NVA06F_SUBCHANNEL_COMPUTE, NV_COMPUTE_NO_OPERATION,
                                     1, NVC46F_DMA_SEC_OP_INC_METHOD);
    pb_words[3] = 0;
    pb_words[4] = push_method_header(NVA06F_SUBCHANNEL_COMPUTE, NV_COMPUTE_PIPE_NOP,
                                     1, NVC46F_DMA_SEC_OP_INC_METHOD);
    pb_words[5] = 0;

    const NvU32 pb_bytes = 6u * sizeof(NvU32);
    const NvU32 entry = *put % 32u;
    write_gpfifo_entry(gp_words, entry, gp_va + pb_offset, pb_bytes);
    __sync_synchronize();
    *put = (entry + 1u) % 32u;
    printf("submit_compute_set_object_pb before GPGet=%u GPPut=%u entry=%u pbVA=0x%016" PRIx64
           " bytes=%u classEngineID=0x%08x\n",
           control[0x88 / 4], control[0x8c / 4], entry, gp_va + pb_offset,
           pb_bytes, class_engine_id);
    control[0x8c / 4] = *put;
    __sync_synchronize();
    usleep(10000);
    printf("submit_compute_set_object_pb after  GPGet=%u GPPut=%u entry0=0x%08x entry1=0x%08x"
           " pb=[0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x]\n",
           control[0x88 / 4], control[0x8c / 4],
           gp_words[entry * 2U], gp_words[entry * 2U + 1U],
           pb_words[0], pb_words[1], pb_words[2], pb_words[3], pb_words[4], pb_words[5]);
}

int main(void)
{
    int rc;
    int ctl_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    int gpu_fd = open("/dev/nvidia0", O_RDWR | O_CLOEXEC);
    int gp_cpu_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    int userd_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    if (ctl_fd < 0 || gpu_fd < 0 || gp_cpu_fd < 0 || userd_fd < 0) {
        perror("open nvidia nodes");
        return 1;
    }

    nv_ioctl_wait_open_complete_t wait_params = {0};
    rc = rm_ioctl_direct(gpu_fd, NV_ESC_WAIT_OPEN_COMPLETE, &wait_params, sizeof(wait_params));
    printf("wait_open rc=%d errno=%d open_rc=%d adapterStatus=0x%08x\n",
           rc, rc == 0 ? 0 : errno, wait_params.rc, wait_params.adapterStatus);

    nv_ioctl_register_fd_t reg_fd = {.ctl_fd = ctl_fd};
    rc = rm_ioctl_direct(gpu_fd, NV_ESC_REGISTER_FD, &reg_fd, sizeof(reg_fd));
    printf("register_fd rc=%d errno=%d ctl_fd=%d\n", rc, rc == 0 ? 0 : errno, ctl_fd);

    nv_ioctl_rm_api_version_t ver = {.cmd = '2'};
    rc = rm_ioctl_direct(ctl_fd, NV_ESC_CHECK_VERSION_STR, &ver, sizeof(ver));
    printf("version_query rc=%d errno=%d reply=%u version=\"%s\"\n",
           rc, rc == 0 ? 0 : errno, ver.reply, ver.versionString);

    nv_ioctl_card_info_t cards[32];
    memset(cards, 0, sizeof(cards));
    rc = rm_ioctl_direct(ctl_fd, NV_ESC_CARD_INFO, cards, sizeof(cards));
    printf("card_info rc=%d errno=%d\n", rc, rc == 0 ? 0 : errno);
    NvU32 gpu_id = 0;
    for (size_t i = 0; i < sizeof(cards) / sizeof(cards[0]); i++) {
        if (!cards[i].valid) {
            continue;
        }
        gpu_id = cards[i].gpu_id;
        printf("  card[%zu] gpu_id=0x%08x pci=%04x:%02x:%02x.%u vendor=0x%04x device=0x%04x minor=%u\n",
               i, cards[i].gpu_id, cards[i].pci_info.domain, cards[i].pci_info.bus,
               cards[i].pci_info.slot, cards[i].pci_info.function, cards[i].pci_info.vendor_id,
               cards[i].pci_info.device_id, cards[i].minor_number);
    }

    if (gpu_id != 0) {
        NvU32 attach[1] = {gpu_id};
        rc = rm_ioctl_direct(ctl_fd, NV_ESC_ATTACH_GPUS_TO_FD, attach, sizeof(attach));
        printf("attach_gpus rc=%d errno=%d gpu_id=0x%08x\n", rc, rc == 0 ? 0 : errno, gpu_id);
    }

    NvHandle client = alloc_client(ctl_fd);
    if (client == 0) {
        fprintf(stderr, "cannot allocate RM client\n");
        return 2;
    }

    NvHandle device = 0;
    NvHandle subdevice = 0;
    if (setup_device(ctl_fd, client, &device, &subdevice) != 0) {
        fprintf(stderr, "cannot allocate RM device\n");
        (void)rm_free(ctl_fd, client, client, client, "free_client");
        return 3;
    }

    NvHandle vaspace = 0;
    (void)allocate_vaspace(ctl_fd, client, device, &vaspace);
    NvHandle dma_for_maps = vaspace;
    NvHandle legacy_ctxdma = 0;
    if (dma_for_maps == 0) {
        fprintf(stderr, "no NV01_MEMORY_VIRTUAL object; GPU VA mapping will likely fail\n");
    }

    const size_t page = 4096;
    const NvU32 sys_flags = NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS |
                            NVOS02_FLAGS_LOCATION_PCI |
                            NVOS02_FLAGS_COHERENCY_WRITE_COMBINE |
                            NVOS02_FLAGS_GPU_CACHEABLE_YES |
                            NVOS02_FLAGS_MAPPING_NO_MAP;
    const NvU32 coherent_flags = NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS |
                                 NVOS02_FLAGS_LOCATION_PCI |
                                 NVOS02_FLAGS_COHERENCY_WRITE_BACK |
                                 NVOS02_FLAGS_GPU_CACHEABLE_YES |
                                 NVOS02_FLAGS_MAPPING_NO_MAP;
    const NvU32 dma_flags = NVOS46_FLAGS_PAGE_SIZE_4KB | NVOS46_FLAGS_CACHE_SNOOP_ENABLE;

    NvHandle notifier = 0x6100;
    NvU64 notifier_va = 0;
    NvHandle error_ctxdma = 0;
    if (rm_alloc_memory(gpu_fd, ctl_fd, client, device, &notifier, NV01_MEMORY_SYSTEM,
                        coherent_flags, page, "alloc_notifier_sysmem") == 0 &&
        dma_for_maps != 0 &&
        try_map_dma_two_devices(ctl_fd, client, device, subdevice, dma_for_maps, notifier,
                                page, dma_flags, &notifier_va, "map_notifier_dma") == 0) {
        error_ctxdma = 0x6200;
        if (allocate_ctxdma(ctl_fd, client, device, subdevice, notifier, 0xff,
                            NVOS03_FLAGS_MAPPING_KERNEL | NVOS03_FLAGS_HASH_TABLE_DISABLE,
                            &error_ctxdma, "alloc_error_notifier_ctxdma") != 0) {
            error_ctxdma = 0;
        }
    }

    NvHandle gpfifo = 0x6300;
    NvP64 gp_linear = 0;
    void *gp_cpu = NULL;
    NvU64 gp_va = 0;
    if (rm_alloc_memory(gpu_fd, gp_cpu_fd, client, device, &gpfifo, NV01_MEMORY_SYSTEM,
                        sys_flags, page, "alloc_gpfifo_sysmem") == 0 &&
        rm_map_memory(ctl_fd, gp_cpu_fd, client, device, gpfifo, page,
                      &gp_linear, "map_gpfifo_cpu") == 0) {
        gp_cpu = map_cpu_from_fd(gp_cpu_fd, page, "gpfifo_cpu");
        if (gp_cpu != NULL) {
            memset(gp_cpu, 0, page);
        }
        if (dma_for_maps != 0) {
            if (try_map_dma_two_devices(ctl_fd, client, device, subdevice, dma_for_maps, gpfifo,
                                        page, dma_flags, &gp_va, "map_gpfifo_dma") != 0) {
                gp_va = 0;
            }
        }
    }

    if (gp_va == 0 && vaspace != 0) {
        legacy_ctxdma = 0x5100;
        if (allocate_ctxdma(ctl_fd, client, device, subdevice, vaspace, 0xffffffffffffffffull,
                            NVOS03_FLAGS_HASH_TABLE_DISABLE, &legacy_ctxdma,
                            "alloc_legacy_va_ctxdma") == 0) {
            dma_for_maps = legacy_ctxdma;
            notifier_va = 0;
            if (notifier != 0) {
                (void)try_map_dma_two_devices(ctl_fd, client, device, subdevice, dma_for_maps,
                                              notifier, page, dma_flags, &notifier_va,
                                              "map_notifier_dma_legacy");
            }
            if (gpfifo != 0) {
                (void)try_map_dma_two_devices(ctl_fd, client, device, subdevice, dma_for_maps,
                                              gpfifo, page, dma_flags, &gp_va,
                                              "map_gpfifo_dma_legacy");
            }
        } else {
            legacy_ctxdma = 0;
        }
    }

    NvHandle userd = 0x6400;
    NvP64 userd_linear = 0;
    void *userd_cpu = NULL;
    if (rm_alloc_memory(gpu_fd, userd_fd, client, device, &userd, NV01_MEMORY_SYSTEM,
                        sys_flags, page, "alloc_userd_sysmem") == 0 &&
        rm_map_memory(ctl_fd, userd_fd, client, device, userd, page,
                      &userd_linear, "map_userd_sysmem_cpu") == 0) {
        userd_cpu = map_cpu_from_fd(userd_fd, page, "userd_sysmem_cpu");
        if (userd_cpu != NULL) {
            memset(userd_cpu, 0, page);
        }
    } else {
        userd = 0;
    }

    NvHandle channel = 0;
    NvHandle channel_userd = 0;
    NvU32 channel_class = 0;
    NvHandle compute = 0;
    NvU32 compute_class = 0;
    NvU32 compute_class_engine_id = 0;
    if (gp_va != 0) {
        if (userd != 0 &&
            try_channel_alloc(ctl_fd, client, device, dma_for_maps,
                              legacy_ctxdma == 0 ? vaspace : 0,
                              error_ctxdma, userd, gp_va, &channel, &channel_class) == 0) {
            channel_userd = userd;
        } else if (try_channel_alloc(ctl_fd, client, device, dma_for_maps,
                                     legacy_ctxdma == 0 ? vaspace : 0,
                                     error_ctxdma, 0, gp_va, &channel, &channel_class) == 0) {
            channel_userd = 0;
        }
    } else {
        printf("channel_alloc skipped: error_ctxdma=0x%x gp_va=0x%016" PRIx64 "\n",
               error_ctxdma, gp_va);
    }

    NvP64 channel_control_linear = 0;
    void *channel_control_cpu = NULL;
    if (channel != 0) {
        NVA06F_CTRL_BIND_PARAMS bind = {.engineType = NV2080_ENGINE_TYPE_GRAPHICS};
        (void)rm_control(ctl_fd, client, channel, NVA06F_CTRL_CMD_BIND, &bind,
                         sizeof(bind), "control_channel_bind_graphics");
        NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS sched = {.bEnable = 1, .bSkipSubmit = 0, .bSkipEnable = 0};
        (void)rm_control(ctl_fd, client, channel, NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, &sched,
                         sizeof(sched), "control_channel_schedule");
        NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS token = {0};
        (void)rm_control(ctl_fd, client, channel, NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
                         &token, sizeof(token), "control_channel_work_submit_token");
        printf("  work_submit_token=0x%08x\n", token.workSubmitToken);
        if (try_compute_alloc(ctl_fd, client, channel, &compute, &compute_class,
                              &compute_class_engine_id) != 0) {
            printf("compute_object_probe skipped: no supported compute class allocated\n");
        }

        void *control_cpu = NULL;
        if (channel_userd != 0) {
            control_cpu = userd_cpu;
            printf("using_client_userd_control hUserd=0x%x cpu=%p\n", channel_userd, control_cpu);
        } else {
            if (rm_map_memory(ctl_fd, userd_fd, client, subdevice != 0 ? subdevice : device,
                              channel, 512, &channel_control_linear, "map_channel_control_subdevice") != 0) {
                (void)rm_map_memory(ctl_fd, userd_fd, client, device,
                                    channel, 512, &channel_control_linear, "map_channel_control_device");
            }
            if (channel_control_linear == 0) {
                if (rm_map_memory(ctl_fd, userd_fd, client, subdevice != 0 ? subdevice : device,
                                  channel, page, &channel_control_linear, "map_channel_control_subdevice_4k") != 0) {
                    (void)rm_map_memory(ctl_fd, userd_fd, client, device,
                                        channel, page, &channel_control_linear, "map_channel_control_device_4k");
                }
            }
            if (channel_control_linear != 0) {
                channel_control_cpu = map_cpu_from_fd(userd_fd, page, "channel_control");
                control_cpu = channel_control_cpu;
            }
        }

        if (gp_cpu != NULL && control_cpu != NULL) {
            volatile NvU32 *gp_words = (volatile NvU32 *)gp_cpu;
            volatile NvU32 *control = (volatile NvU32 *)control_cpu;
            NvU32 put = control[0x8c / 4] % 32u;
            NvU32 entry = put;
            gp_words[entry * 2U] = 0;
            gp_words[entry * 2U + 1U] = 0;
            __sync_synchronize();
            printf("submit_gpfifo_nop before GPGet=%u GPPut=%u entry=%u\n",
                   control[0x88 / 4], control[0x8c / 4], entry);
            put = (entry + 1u) % 32u;
            control[0x8c / 4] = put;
            __sync_synchronize();
            usleep(10000);
            printf("submit_gpfifo_nop after  GPGet=%u GPPut=%u entry0=0x%08x entry1=0x%08x\n",
                   control[0x88 / 4], control[0x8c / 4],
                   gp_words[entry * 2U], gp_words[entry * 2U + 1U]);
            if (compute != 0 && compute_class_engine_id != 0) {
                submit_compute_set_object_pb(gp_words, control, gp_va, &put,
                                             compute_class_engine_id);
            } else {
                printf("submit_compute_set_object_pb skipped: compute=0x%x classEngineID=0x%08x\n",
                       compute, compute_class_engine_id);
            }
        } else {
            printf("submit_gpfifo_nop skipped: gp_cpu=%p control_cpu=%p\n", gp_cpu, control_cpu);
        }
    }

    if (compute != 0) {
        (void)rm_free(ctl_fd, client, channel, compute, "free_compute");
    }
    if (channel_control_cpu != NULL) {
        munmap(channel_control_cpu, page);
    }
    (void)rm_unmap_memory(ctl_fd, client, subdevice != 0 ? subdevice : device,
                          channel, channel_control_linear, "unmap_channel_control");
    if (channel != 0) {
        (void)rm_free(ctl_fd, client, device, channel, "free_channel");
    }
    if (userd_cpu != NULL) {
        munmap(userd_cpu, page);
    }
    (void)rm_unmap_memory(ctl_fd, client, device, userd, userd_linear, "unmap_userd_sysmem_cpu");
    if (userd != 0) {
        (void)rm_free(ctl_fd, client, device, userd, "free_userd");
    }
    if (gp_va != 0) {
        (void)rm_unmap_memory_dma(ctl_fd, client, device, dma_for_maps, gpfifo, gp_va,
                                  page, "unmap_gpfifo_dma");
    }
    if (gp_cpu != NULL) {
        munmap(gp_cpu, page);
    }
    (void)rm_unmap_memory(ctl_fd, client, device, gpfifo, gp_linear, "unmap_gpfifo_cpu");
    if (gpfifo != 0) {
        (void)rm_free(ctl_fd, client, device, gpfifo, "free_gpfifo");
    }
    if (error_ctxdma != 0) {
        (void)rm_free(ctl_fd, client, device, error_ctxdma, "free_error_ctxdma");
    }
    if (notifier_va != 0) {
        (void)rm_unmap_memory_dma(ctl_fd, client, device, dma_for_maps, notifier,
                                  notifier_va, page, "unmap_notifier_dma");
    }
    if (notifier != 0) {
        (void)rm_free(ctl_fd, client, device, notifier, "free_notifier");
    }
    if (legacy_ctxdma != 0) {
        (void)rm_free(ctl_fd, client, device, legacy_ctxdma, "free_legacy_va_ctxdma");
    }
    if (vaspace != 0) {
        (void)rm_free(ctl_fd, client, device, vaspace, "free_vaspace");
    }
    if (subdevice != 0) {
        (void)rm_free(ctl_fd, client, device, subdevice, "free_subdevice");
    }
    (void)rm_free(ctl_fd, client, client, device, "free_device");
    (void)rm_free(ctl_fd, client, client, client, "free_client");

    close(userd_fd);
    close(gp_cpu_fd);
    close(gpu_fd);
    close(ctl_fd);
    return channel != 0 ? 0 : 4;
}
