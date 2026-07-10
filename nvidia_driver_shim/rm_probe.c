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

#define NV01_NULL_OBJECT 0x0u
#define NV01_ROOT 0x0u
#define NV01_ROOT_CLIENT 0x41u
#define NV01_DEVICE_0 0x80u
#define NV20_SUBDEVICE_0 0x2080u
#define NV01_MEMORY_SYSTEM 0x3eu
#define NV01_MEMORY_LOCAL_USER 0x40u

#define NV_OK 0u

#define NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS (1u << 4)
#define NVOS02_FLAGS_LOCATION_PCI (0u << 8)
#define NVOS02_FLAGS_LOCATION_VIDMEM (2u << 8)
#define NVOS02_FLAGS_COHERENCY_CACHED (1u << 12)
#define NVOS02_FLAGS_COHERENCY_WRITE_COMBINE (2u << 12)
#define NVOS02_FLAGS_GPU_CACHEABLE_YES (1u << 18)
#define NVOS02_FLAGS_MAPPING_NO_MAP (1u << 30)

#define NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES 0u

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
    NvHandle hClient;
    NvHandle hObject;
    NvV32 cmd;
    NvU32 flags;
    NvP64 params __attribute__((aligned(8)));
    NvU32 paramsSize;
    NvV32 status;
} NVOS54_PARAMETERS;

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

int main(void)
{
    int ctl_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    int gpu_fd = open("/dev/nvidia0", O_RDWR | O_CLOEXEC);
    if (ctl_fd < 0 || gpu_fd < 0) {
        perror("open nvidia nodes");
        return 1;
    }

    nv_ioctl_wait_open_complete_t wait_params = {0};
    int rc = rm_ioctl_direct(gpu_fd, NV_ESC_WAIT_OPEN_COMPLETE, &wait_params, sizeof(wait_params));
    printf("wait_open rc=%d errno=%d open_rc=%d adapterStatus=0x%08x\n",
           rc, rc == 0 ? 0 : errno, wait_params.rc, wait_params.adapterStatus);

    nv_ioctl_register_fd_t reg_fd = {.ctl_fd = ctl_fd};
    rc = rm_ioctl_direct(gpu_fd, NV_ESC_REGISTER_FD, &reg_fd, sizeof(reg_fd));
    printf("register_fd rc=%d errno=%d ctl_fd=%d\n", rc, rc == 0 ? 0 : errno, ctl_fd);

    nv_ioctl_rm_api_version_t ver = {.cmd = '2'};
    rc = rm_ioctl_direct(ctl_fd, NV_ESC_CHECK_VERSION_STR, &ver, sizeof(ver));
    printf("version_query rc=%d errno=%d reply=%u version=\"%s\"\n", rc, rc == 0 ? 0 : errno, ver.reply, ver.versionString);

    nv_ioctl_card_info_t cards[32];
    memset(cards, 0, sizeof(cards));
    rc = rm_ioctl_direct(ctl_fd, NV_ESC_CARD_INFO, cards, sizeof(cards));
    printf("card_info rc=%d errno=%d\n", rc, rc == 0 ? 0 : errno);
    NvU32 gpu_id = 0;
    for (size_t i = 0; i < sizeof(cards) / sizeof(cards[0]); i++) {
        if (!cards[i].valid) continue;
        gpu_id = cards[i].gpu_id;
        printf("  card[%zu] gpu_id=0x%08x pci=%04x:%02x:%02x.%u vendor=0x%04x device=0x%04x minor=%u fb=0x%016" PRIx64 "/0x%016" PRIx64 "\n",
               i, cards[i].gpu_id, cards[i].pci_info.domain, cards[i].pci_info.bus,
               cards[i].pci_info.slot, cards[i].pci_info.function, cards[i].pci_info.vendor_id,
               cards[i].pci_info.device_id, cards[i].minor_number, cards[i].fb_address, cards[i].fb_size);
    }

    if (gpu_id != 0) {
        NvU32 attach[1] = {gpu_id};
        rc = rm_ioctl_direct(ctl_fd, NV_ESC_ATTACH_GPUS_TO_FD, attach, sizeof(attach));
        printf("attach_gpus rc=%d errno=%d gpu_id=0x%08x\n", rc, rc == 0 ? 0 : errno, gpu_id);
    }

    NvHandle client = 0;
    NvU32 root_out = 0;
    if (rm_alloc(ctl_fd, NV01_NULL_OBJECT, NV01_NULL_OBJECT, &client, NV01_ROOT, &root_out, sizeof(root_out), "alloc_root:size4") != 0) {
        client = 0;
        root_out = 0;
        (void)rm_alloc(ctl_fd, NV01_NULL_OBJECT, NV01_NULL_OBJECT, &client, NV01_ROOT, &root_out, 0, "alloc_root:size0");
    }
    printf("root_out=0x%x client=0x%x\n", root_out, client);
    if (client == 0 && root_out != 0) {
        client = root_out;
    }
    if (client == 0) {
        fprintf(stderr, "cannot allocate RM client; stopping\n");
        return 2;
    }

    NvHandle device = 0x1000;
    NV0080_ALLOC_PARAMETERS dev_params;
    memset(&dev_params, 0, sizeof(dev_params));
    dev_params.deviceId = 0;
    dev_params.hClientShare = client;
    dev_params.vaMode = NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES;
    if (rm_alloc(ctl_fd, client, client, &device, NV01_DEVICE_0, &dev_params, sizeof(dev_params), "alloc_device") != 0) {
        fprintf(stderr, "cannot allocate RM device; stopping\n");
        (void)rm_free(ctl_fd, client, client, client, "free_client");
        return 3;
    }

    NvHandle subdevice = 0x2000;
    NV2080_ALLOC_PARAMETERS sub_params = {.subDeviceId = 0};
    if (rm_alloc(ctl_fd, client, device, &subdevice, NV20_SUBDEVICE_0, &sub_params, sizeof(sub_params), "alloc_subdevice") != 0) {
        fprintf(stderr, "subdevice allocation failed; continuing with device memory probe\n");
    }

    size_t size = 4096;
    NvHandle sysmem = 0x3000;
    NvU32 sys_flags = NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS |
                      NVOS02_FLAGS_LOCATION_PCI |
                      NVOS02_FLAGS_COHERENCY_WRITE_COMBINE |
                      NVOS02_FLAGS_GPU_CACHEABLE_YES |
                      NVOS02_FLAGS_MAPPING_NO_MAP;
    if (rm_alloc_memory(gpu_fd, ctl_fd, client, device, &sysmem, NV01_MEMORY_SYSTEM, sys_flags, size, "alloc_memory_system") == 0) {
        NvP64 linear = 0;
        if (rm_map_memory(ctl_fd, ctl_fd, client, device, sysmem, size, &linear, "map_memory_system_ctlfd") == 0) {
            void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, ctl_fd, 0);
            printf("  mmap ctl fd -> %p errno=%d:%s\n", addr, addr == MAP_FAILED ? errno : 0, addr == MAP_FAILED ? strerror(errno) : "ok");
            if (addr != MAP_FAILED) {
                volatile uint32_t *p = (volatile uint32_t *)addr;
                p[0] = 0x13572468u;
                printf("  mmap write/read 0x%08x\n", p[0]);
                munmap(addr, size);
            }
            rm_unmap_memory(ctl_fd, client, device, sysmem, linear, "unmap_memory_system");
        }
        rm_free(ctl_fd, client, device, sysmem, "free_sysmem");
    }

    NvHandle vidmem = 0x4000;
    NvU32 vid_flags = NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS |
                      NVOS02_FLAGS_LOCATION_VIDMEM |
                      NVOS02_FLAGS_COHERENCY_WRITE_COMBINE |
                      NVOS02_FLAGS_GPU_CACHEABLE_YES |
                      NVOS02_FLAGS_MAPPING_NO_MAP;
    if (rm_alloc_memory(gpu_fd, gpu_fd, client, device, &vidmem, NV01_MEMORY_LOCAL_USER, vid_flags, size, "alloc_memory_vidmem") == 0) {
        NvP64 linear = 0;
        if (rm_map_memory(ctl_fd, gpu_fd, client, device, vidmem, size, &linear, "map_memory_vidmem_gpufd") == 0) {
            void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, gpu_fd, 0);
            printf("  mmap gpu fd -> %p errno=%d:%s\n", addr, addr == MAP_FAILED ? errno : 0, addr == MAP_FAILED ? strerror(errno) : "ok");
            if (addr != MAP_FAILED) {
                volatile uint32_t *p = (volatile uint32_t *)addr;
                p[0] = 0x24681357u;
                printf("  mmap write/read 0x%08x\n", p[0]);
                munmap(addr, size);
            }
            rm_unmap_memory(ctl_fd, client, device, vidmem, linear, "unmap_memory_vidmem");
        }
        rm_free(ctl_fd, client, device, vidmem, "free_vidmem");
    }

    if (subdevice != 0x2000) {
        (void)rm_free(ctl_fd, client, device, subdevice, "free_subdevice");
    } else {
        (void)rm_free(ctl_fd, client, device, 0x2000, "free_subdevice");
    }
    (void)rm_free(ctl_fd, client, client, device, "free_device");
    (void)rm_free(ctl_fd, client, client, client, "free_client");

    close(gpu_fd);
    close(ctl_fd);
    return 0;
}
