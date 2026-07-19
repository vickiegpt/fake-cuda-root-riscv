#define _GNU_SOURCE
#include "../include/cuda.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

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
#undef cuTensorMapEncodeTiled
#undef cuTensorMapEncodeIm2col
#undef cuTensorMapEncodeIm2colWide
#undef cuTensorMapReplaceAddress

#define LANXIN_CUDA_VERSION 12090
#define DEFAULT_TOTAL_MEM_MB 32768ULL
#define DEFAULT_SM_COUNT 170
#define DEFAULT_COMPUTE_MAJOR 12
#define DEFAULT_COMPUTE_MINOR 0
#define DEFAULT_ACCOUNTING_DIR "/tmp/lanxin_nvidia_cuda_accounting"
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
#define RM_BLACKWELL_USERMODE_A 0xc761u
#define RM_HOPPER_USERMODE_A 0xc661u
#define RM_VOLTA_USERMODE_A 0xc361u
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
#define RM_NVOS33_FLAGS_ACCESS_WRITE_ONLY 0x2u
#define RM_NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES 0u
#define RM_NVA06F_CTRL_CMD_GPFIFO_SCHEDULE 0xa06f0103u
#define RM_NVA06F_CTRL_CMD_BIND 0xa06f0104u
#define RM_NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN 0xc36f0108u
#define RM_NVC36F_CTRL_CMD_GPFIFO_SET_WORK_SUBMIT_TOKEN_NOTIF_INDEX 0xc36f010au
#define RM_NV906F_CTRL_GET_CLASS_ENGINEID 0x906f0101u
#define RM_NV_CHANNELGPFIFO_NOTIFICATION_TYPE_SIZE_1 3u
#define RM_NV_NOTIFICATION_BYTES 16u
#define RM_NV_NOTIFICATION_INFO32_OFFSET 8u
#define RM_NVC361_NV_USERMODE_SIZE 65536u
#define RM_NVC361_NOTIFY_CHANNEL_PENDING 0x90u
#define RM_NVA06F_SUBCHANNEL_COMPUTE 1u
#define RM_NVC46F_DMA_SEC_OP_INC_METHOD 1u
#define RM_COMPUTE_SET_OBJECT 0x0000u
#define RM_COMPUTE_NO_OPERATION 0x0100u
#define RM_COMPUTE_PIPE_NOP 0x1a2cu
#define RM_COMPUTE_SET_PROGRAM_REGION_A 0x1608u
#define RM_COMPUTE_SET_PROGRAM_REGION_B 0x160cu
#define RM_COMPUTE_SET_QMD_VERSION 0x0288u
#define RM_COMPUTE_CHECK_QMD_VERSION 0x0290u
#define RM_COMPUTE_SET_CWD_SLOT_COUNT 0x02b0u
#define RM_COMPUTE_SEND_PCAS_A 0x02b4u
#define RM_COMPUTE_SEND_PCAS_B 0x02b8u
#define RM_COMPUTE_SEND_SIGNALING_PCAS_B 0x02bcu
#define RM_CHANNEL_SUBCHANNEL_HOST 0u
#define RM_CHANNEL_SEM_ADDR_LO 0x005cu
#define RM_CHANNEL_SEM_EXECUTE_RELEASE (1u << 0)
#define RM_CHANNEL_SEM_EXECUTE_RELEASE_WFI_EN (1u << 20)
#define RM_CHANNEL_SEM_EXECUTE_PAYLOAD_SIZE_32BIT 0u
#define RM_CHANNEL_SEM_EXECUTE_PAYLOAD_SIZE_64BIT (1u << 24)
#define RM_CHANNEL_SEM_EXECUTE_RELEASE_TIMESTAMP_DIS 0u
#define LANXIN_HW_QMD_WORDS 64u
#define LANXIN_HW_QMD_BYTES (LANXIN_HW_QMD_WORDS * sizeof(rm_u32))
#define LANXIN_QMD_VERSION_01_06 0x0106u
#define LANXIN_QMD_MAJOR_VERSION 1u
#define LANXIN_QMD_MINOR_VERSION 6u
#define LANXIN_FATBINC_MAGIC 0x466243b1u
#define LANXIN_QMD_STAGING_MAGIC 0x4c584e56514d4431ULL
#define LANXIN_QMD_STAGING_VERSION 1u
#define LANXIN_COMPLETION_MAGIC 0x4c584e56434d5031ULL
#define LANXIN_COMPLETION_PENDING 1u
#define LANXIN_COMPLETION_DONE 2u
#define LANXIN_COMPLETION_TIMEOUT 3u
#define LANXIN_LAUNCH_PARAM_EXTRA_BUFFER 1u
#define LANXIN_LAUNCH_PARAM_POINTER_ARRAY 2u
#define LANXIN_CODE_STAGE_WHOLE_IMAGE 0u
#define LANXIN_CODE_STAGE_TEXT_ONLY 1u
#define LANXIN_ELF_EM_CUDA 190u
#define LANXIN_ELF_SHT_PROGBITS 1u
#define LANXIN_ELF_SHT_SYMTAB 2u
#define LANXIN_ELF_SHT_STRTAB 3u
#define LANXIN_ELF_SHT_NOBITS 8u
#define LANXIN_ELF_SHT_CUDA_INFO 0x70000000u
#define LANXIN_ELF_SHF_ALLOC 0x2ULL
#define LANXIN_ELF_SHF_EXECINSTR 0x4ULL
#define LANXIN_ELF_STT_FUNC 2u
#define LANXIN_ELF_STT_CUDA_FUNC 10u
#define LANXIN_EIATTR_REGCOUNT 0x2f04u
#define LANXIN_EIATTR_REGCOUNT_ALT 0x0401u
#define LANXIN_EIATTR_MAX_THREADS 0x0504u
#define LANXIN_EIATTR_SMEM_SIZE 0x0808u
#define LANXIN_EIATTR_LMEM_SIZE 0x0a04u

#define LANXIN_TORCH_DTYPE_BYTE 0
#define LANXIN_TORCH_DTYPE_CHAR 1
#define LANXIN_TORCH_DTYPE_SHORT 2
#define LANXIN_TORCH_DTYPE_INT 3
#define LANXIN_TORCH_DTYPE_LONG 4
#define LANXIN_TORCH_DTYPE_HALF 5
#define LANXIN_TORCH_DTYPE_FLOAT 6
#define LANXIN_TORCH_DTYPE_DOUBLE 7
#define LANXIN_TORCH_DTYPE_BOOL 11
#define LANXIN_TORCH_DTYPE_BFLOAT16 15
#define LANXIN_TORCH_DTYPE_UINT16 27
#define LANXIN_TORCH_DTYPE_UINT32 28
#define LANXIN_TORCH_DTYPE_UINT64 29

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
    rm_u32 index;
} rm_channel_token_notif_index_params_t;

typedef struct {
    rm_bool bBar1Mapping;
    rm_bool bPriv;
} rm_hopper_usermode_params_t;

typedef struct {
    rm_handle hObject;
    rm_u32 classEngineID;
    rm_u32 classID;
    rm_u32 engineID;
} rm_get_class_engine_id_params_t;

typedef struct {
    int magic;
    int version;
    const unsigned long long *data;
    void *filename_or_fatbins;
} lanxin_fatbin_wrapper_t;

struct rm_stage_buffer {
    rm_handle memory;
    rm_p64 linear;
    rm_u64 gpu_va;
    void *cpu;
    size_t size;
    int map_fd;
};

struct lanxin_kernel_metadata {
    bool valid;
    rm_u64 text_file_offset;
    rm_u64 text_size;
    rm_u64 text_addr;
    rm_u64 symbol_value;
    rm_u64 symbol_size;
    rm_u64 const0_file_offset;
    rm_u64 const0_size;
    rm_u32 qmd_program_offset;
    rm_u32 reg_count;
    rm_u32 static_shared_mem_bytes;
    rm_u32 local_mem_bytes;
    rm_u32 const_mem_bytes;
    rm_u32 max_threads_per_block;
    rm_u32 sm_version;
};

struct lanxin_qmd_staging_desc {
    rm_u64 magic;
    rm_u32 version;
    rm_u32 size;
    rm_u64 launch_id;
    rm_u32 grid[3];
    rm_u32 block[3];
    rm_u32 shared_mem_bytes;
    rm_u32 flags;
    rm_u64 code_va;
    rm_u64 code_bytes;
    rm_u64 qmd_va;
    rm_u64 params_va;
    rm_u64 params_bytes;
    rm_u64 completion_va;
    rm_u64 module_image_bytes;
    rm_u64 module_image_hash;
    rm_u64 function_name_hash;
    rm_u32 compute_class;
    rm_u32 class_engine_id;
    rm_u32 param_flags;
    rm_u32 metadata_flags;
    rm_u64 text_file_offset;
    rm_u64 text_size;
    rm_u64 const0_file_offset;
    rm_u64 const0_size;
    rm_u32 qmd_program_offset;
    rm_u32 reg_count;
    rm_u32 static_shared_mem_bytes;
    rm_u32 local_mem_bytes;
    rm_u32 const_mem_bytes;
    rm_u32 max_threads_per_block;
    rm_u32 sm_version;
    rm_u32 reserved0;
    rm_u64 reserved[4];
};

struct lanxin_launch_completion {
    volatile rm_u64 host_progress;
    volatile rm_u32 qmd_done;
    rm_u32 status;
    rm_u64 magic;
    rm_u64 launch_id;
    rm_u64 gpfifo_put;
    rm_u64 timestamp_ns;
    rm_u64 expected_host_progress;
    rm_u32 expected_qmd_done;
    rm_u32 qmd_submitted;
    rm_u64 reserved[4];
};

struct launch_staging {
    struct rm_stage_buffer pushbuffer;
    struct rm_stage_buffer qmd;
    struct rm_stage_buffer params;
    struct rm_stage_buffer completion;
    unsigned long long launch_id;
};

struct launch_request {
    rm_u32 grid[3];
    rm_u32 block[3];
    rm_u32 shared_mem_bytes;
    const void *params;
    size_t params_size;
    rm_u32 param_flags;
};

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
    bool kernels_discovered;
    struct rm_stage_buffer code;
    size_t staged_code_bytes;
    rm_u32 staged_code_layout;
    rm_u64 staged_text_file_offset;
    rm_u64 staged_text_size;
    rm_u64 staged_function_name_hash;
    rm_u64 image_hash;
    struct CUfunc_st *functions;
    struct module_global *globals;
    struct CUmod_st *next;
};

struct CUfunc_st {
    uint32_t magic;
    char *name;
    CUmodule module;
    rm_u64 name_hash;
    struct lanxin_kernel_metadata metadata;
    struct launch_staging launch;
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
    int rm_notifier_fd;
    int rm_gpfifo_fd;
    int rm_userd_fd;
    int rm_usermode_fd;
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
    rm_handle rm_usermode;
    rm_handle rm_usermode_parent;
    rm_handle rm_channel;
    rm_handle rm_compute;
    rm_p64 rm_notifier_va;
    rm_p64 rm_notifier_linear;
    rm_p64 rm_gpfifo_va;
    rm_p64 rm_gpfifo_linear;
    rm_p64 rm_userd_linear;
    rm_p64 rm_usermode_linear;
    size_t rm_gpfifo_size;
    void *rm_notifier_cpu;
    void *rm_gpfifo_cpu;
    void *rm_userd_cpu;
    void *rm_usermode_cpu;
    rm_u32 rm_work_submit_token;
    rm_u32 rm_doorbell_token;
    rm_u32 rm_token_notifier_index;
    rm_u32 rm_gpfifo_put;
    rm_u32 rm_usermode_class;
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
    .rm_notifier_fd = -1,
    .rm_gpfifo_fd = -1,
    .rm_userd_fd = -1,
    .rm_usermode_fd = -1,
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

static const char *accounting_dir(void)
{
    const char *dir = getenv("LANXIN_NVIDIA_CUDA_ACCOUNTING_DIR");
    return (dir != NULL && dir[0] != '\0') ? dir : DEFAULT_ACCOUNTING_DIR;
}

static void update_process_memory_accounting_locked(void)
{
    const char *dir = accounting_dir();
    if (mkdir(dir, 0777) != 0 && errno != EEXIST) {
        return;
    }

    char path[PATH_MAX];
    int written = snprintf(path, sizeof(path), "%s/%ld.mem", dir, (long)getpid());
    if (written <= 0 || (size_t)written >= sizeof(path)) {
        return;
    }

    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return;
    }
    fprintf(fp, "%llu\n", (unsigned long long)g.allocated_bytes);
    fclose(fp);
}

__attribute__((destructor)) static void cleanup_process_memory_accounting(void)
{
    const char *dir = accounting_dir();
    char path[PATH_MAX];
    int written = snprintf(path, sizeof(path), "%s/%ld.mem", dir, (long)getpid());
    if (written > 0 && (size_t)written < sizeof(path)) {
        unlink(path);
    }
}

static __thread CUcontext tls_current_ctx;
static __thread CUcontext tls_ctx_stack[32];
static __thread int tls_ctx_depth;

static bool valid_module(CUmodule module);
static bool valid_function(CUfunction function);
static rm_u32 env_u32(const char *name, rm_u32 fallback);

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

static rm_u64 fnv1a64_bytes(const void *data, size_t size)
{
    const unsigned char *bytes = (const unsigned char *)data;
    rm_u64 hash = 1469598103934665603ULL;
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static rm_u64 fnv1a64_str(const char *text)
{
    return text != NULL ? fnv1a64_bytes(text, strlen(text)) : 0;
}

static uint16_t rd16le(const unsigned char *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static rm_u32 rd32le(const unsigned char *p)
{
    return (rm_u32)p[0] | ((rm_u32)p[1] << 8) |
           ((rm_u32)p[2] << 16) | ((rm_u32)p[3] << 24);
}

static rm_u64 rd64le(const unsigned char *p)
{
    return (rm_u64)rd32le(p) | ((rm_u64)rd32le(p + 4) << 32);
}

static size_t image_scan_limit(void)
{
    unsigned long long limit = env_ull("LANXIN_NVIDIA_CUDA_IMAGE_SCAN_MAX_BYTES",
                                       64ULL * 1024ULL * 1024ULL);
    if (limit == 0) {
        limit = 1;
    }
    return limit > (unsigned long long)SIZE_MAX ? SIZE_MAX : (size_t)limit;
}

static size_t probe_elf_image_size(const unsigned char *bytes)
{
    if (bytes == NULL ||
        bytes[0] != 0x7f || bytes[1] != 'E' || bytes[2] != 'L' || bytes[3] != 'F') {
        return 0;
    }
    if (bytes[4] != 1 && bytes[4] != 2) {
        return 4;
    }
    if (bytes[5] != 1) {
        return 4;
    }

    size_t limit = image_scan_limit();
    size_t max_end = bytes[4] == 2 ? 64U : 52U;
    if (bytes[4] == 2) {
        rm_u64 phoff = rd64le(bytes + 32);
        rm_u64 shoff = rd64le(bytes + 40);
        uint16_t phentsize = rd16le(bytes + 54);
        uint16_t phnum = rd16le(bytes + 56);
        uint16_t shentsize = rd16le(bytes + 58);
        uint16_t shnum = rd16le(bytes + 60);
        if (shentsize >= 64 && shoff < limit && shnum < 8192) {
            rm_u64 shdr_bytes = (rm_u64)shentsize * shnum;
            if (shdr_bytes <= limit - shoff && shoff + shdr_bytes > max_end) {
                max_end = (size_t)(shoff + shdr_bytes);
            }
        }
        if (phentsize >= 56 && phoff < limit && phnum < 8192) {
            for (uint16_t i = 0; i < phnum; i++) {
                rm_u64 off = phoff + (rm_u64)i * phentsize;
                if (off + 56 > limit) {
                    break;
                }
                rm_u64 p_offset = rd64le(bytes + off + 8);
                rm_u64 p_filesz = rd64le(bytes + off + 32);
                if (p_offset <= limit && p_filesz <= limit - p_offset &&
                    p_offset + p_filesz > max_end) {
                    max_end = (size_t)(p_offset + p_filesz);
                }
            }
        }
        if (shentsize >= 64 && shoff < limit && shnum < 8192) {
            for (uint16_t i = 0; i < shnum; i++) {
                rm_u64 off = shoff + (rm_u64)i * shentsize;
                if (off + 64 > limit) {
                    break;
                }
                rm_u32 sh_type = rd32le(bytes + off + 4);
                rm_u64 sh_offset = rd64le(bytes + off + 24);
                rm_u64 sh_size = rd64le(bytes + off + 32);
                if (sh_type != 8 && sh_offset <= limit && sh_size <= limit - sh_offset &&
                    sh_offset + sh_size > max_end) {
                    max_end = (size_t)(sh_offset + sh_size);
                }
            }
        }
    } else {
        rm_u32 phoff = rd32le(bytes + 28);
        rm_u32 shoff = rd32le(bytes + 32);
        uint16_t phentsize = rd16le(bytes + 42);
        uint16_t phnum = rd16le(bytes + 44);
        uint16_t shentsize = rd16le(bytes + 46);
        uint16_t shnum = rd16le(bytes + 48);
        if (shentsize >= 40 && shoff < limit && shnum < 8192) {
            rm_u64 shdr_bytes = (rm_u64)shentsize * shnum;
            if (shdr_bytes <= limit - shoff && shoff + shdr_bytes > max_end) {
                max_end = (size_t)(shoff + shdr_bytes);
            }
        }
        if (phentsize >= 32 && phoff < limit && phnum < 8192) {
            for (uint16_t i = 0; i < phnum; i++) {
                rm_u64 off = (rm_u64)phoff + (rm_u64)i * phentsize;
                if (off + 32 > limit) {
                    break;
                }
                rm_u32 p_offset = rd32le(bytes + off + 4);
                rm_u32 p_filesz = rd32le(bytes + off + 16);
                if ((rm_u64)p_offset + p_filesz <= limit &&
                    (size_t)p_offset + p_filesz > max_end) {
                    max_end = (size_t)p_offset + p_filesz;
                }
            }
        }
        if (shentsize >= 40 && shoff < limit && shnum < 8192) {
            for (uint16_t i = 0; i < shnum; i++) {
                rm_u64 off = (rm_u64)shoff + (rm_u64)i * shentsize;
                if (off + 40 > limit) {
                    break;
                }
                rm_u32 sh_type = rd32le(bytes + off + 4);
                rm_u32 sh_offset = rd32le(bytes + off + 16);
                rm_u32 sh_size = rd32le(bytes + off + 20);
                if (sh_type != 8 && (rm_u64)sh_offset + sh_size <= limit &&
                    (size_t)sh_offset + sh_size > max_end) {
                    max_end = (size_t)sh_offset + sh_size;
                }
            }
        }
    }
    return max_end;
}

static const void *resolve_module_image(const void *image, size_t *size_out, const char **kind_out)
{
    const unsigned char *bytes = (const unsigned char *)image;
    const char *kind = "unknown";
    size_t image_size = 0;
    if (image == NULL) {
        if (size_out != NULL) {
            *size_out = 0;
        }
        if (kind_out != NULL) {
            *kind_out = kind;
        }
        return NULL;
    }

    const lanxin_fatbin_wrapper_t *wrapper = (const lanxin_fatbin_wrapper_t *)image;
    if ((rm_u32)wrapper->magic == LANXIN_FATBINC_MAGIC && wrapper->data != NULL) {
        bytes = (const unsigned char *)wrapper->data;
        kind = "fatbin-wrapper";
    }

    image_size = probe_elf_image_size(bytes);
    if (image_size != 0) {
        if (strcmp(kind, "fatbin-wrapper") != 0) {
            kind = "elf";
        }
    } else {
        size_t limit = image_scan_limit();
        image_size = strnlen((const char *)bytes, limit);
        if (image_size != limit) {
            image_size++;
            if (strcmp(kind, "fatbin-wrapper") != 0) {
                kind = "text";
            }
        } else {
            image_size = 0;
            if (strcmp(kind, "fatbin-wrapper") != 0) {
                kind = "raw";
            }
        }
    }

    if (size_out != NULL) {
        *size_out = image_size;
    }
    if (kind_out != NULL) {
        *kind_out = kind;
    }
    return bytes;
}

struct elf64_section_view {
    const unsigned char *header;
    const char *name;
    rm_u32 type;
    rm_u64 flags;
    rm_u64 addr;
    rm_u64 offset;
    rm_u64 size;
    rm_u32 link;
    rm_u32 info;
    rm_u64 entsize;
    uint16_t index;
};

static bool range_in_image(rm_u64 offset, rm_u64 size, size_t image_size)
{
    return offset <= (rm_u64)image_size && size <= (rm_u64)image_size - offset;
}

static const char *elf_string_at(const unsigned char *base, rm_u64 size, rm_u32 offset)
{
    if (base == NULL || offset >= size) {
        return NULL;
    }
    const char *text = (const char *)base + offset;
    size_t remain = (size_t)(size - offset);
    return memchr(text, '\0', remain) != NULL ? text : NULL;
}

static bool elf_section_from_header(const unsigned char *image, size_t image_size,
                                    const unsigned char *shdr, uint16_t index,
                                    const unsigned char *shstr, rm_u64 shstr_size,
                                    struct elf64_section_view *out)
{
    if (image == NULL || shdr == NULL || out == NULL) {
        return false;
    }
    rm_u32 name_off = rd32le(shdr + 0);
    rm_u32 type = rd32le(shdr + 4);
    rm_u64 flags = rd64le(shdr + 8);
    rm_u64 addr = rd64le(shdr + 16);
    rm_u64 offset = rd64le(shdr + 24);
    rm_u64 size = rd64le(shdr + 32);
    rm_u32 link = rd32le(shdr + 40);
    rm_u32 info = rd32le(shdr + 44);
    rm_u64 entsize = rd64le(shdr + 56);
    if (type != LANXIN_ELF_SHT_NOBITS && !range_in_image(offset, size, image_size)) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    out->header = shdr;
    out->name = elf_string_at(shstr, shstr_size, name_off);
    out->type = type;
    out->flags = flags;
    out->addr = addr;
    out->offset = offset;
    out->size = size;
    out->link = link;
    out->info = info;
    out->entsize = entsize;
    out->index = index;
    return true;
}

static bool kernel_text_name_matches(const char *section_name, const char *kernel_name)
{
    const char prefix[] = ".text.";
    if (section_name == NULL || kernel_name == NULL ||
        strncmp(section_name, prefix, sizeof(prefix) - 1U) != 0) {
        return false;
    }
    const char *suffix = section_name + sizeof(prefix) - 1U;
    return strcmp(suffix, kernel_name) == 0;
}

static bool kernel_named_section_matches(const char *section_name, const char *prefix,
                                         const char *kernel_name)
{
    if (section_name == NULL || prefix == NULL || kernel_name == NULL) {
        return false;
    }
    size_t prefix_len = strlen(prefix);
    return strncmp(section_name, prefix, prefix_len) == 0 &&
           strcmp(section_name + prefix_len, kernel_name) == 0;
}

static const char *kernel_name_from_text_section(const char *section_name)
{
    const char prefix[] = ".text.";
    if (section_name == NULL ||
        strncmp(section_name, prefix, sizeof(prefix) - 1U) != 0) {
        return NULL;
    }
    const char *name = section_name + sizeof(prefix) - 1U;
    return name[0] != '\0' ? name : NULL;
}

static void parse_kernel_nv_info(const unsigned char *data, rm_u64 size,
                                 struct lanxin_kernel_metadata *meta)
{
    if (data == NULL || meta == NULL) {
        return;
    }
    rm_u64 off = 0;
    while (off + 4 <= size) {
        rm_u32 attr_type = rd16le(data + off);
        rm_u32 attr_size = rd16le(data + off + 2);
        off += 4;
        if (attr_size > size - off) {
            break;
        }
        const unsigned char *payload = data + off;
        if (attr_size >= 4) {
            rm_u32 value = rd32le(payload);
            switch (attr_type) {
            case LANXIN_EIATTR_REGCOUNT:
            case LANXIN_EIATTR_REGCOUNT_ALT:
                meta->reg_count = value;
                break;
            case LANXIN_EIATTR_MAX_THREADS:
                meta->max_threads_per_block = value;
                break;
            case LANXIN_EIATTR_SMEM_SIZE:
                meta->static_shared_mem_bytes = value;
                break;
            case LANXIN_EIATTR_LMEM_SIZE:
                meta->local_mem_bytes = value;
                break;
            default:
                break;
            }
        }
        off += attr_size;
        off = (off + 3ULL) & ~3ULL;
    }
}

static bool module_find_kernel_metadata(CUmodule module, const char *kernel_name,
                                        struct lanxin_kernel_metadata *meta)
{
    if (!valid_module(module) || kernel_name == NULL || meta == NULL ||
        module->image == NULL || module->image_size < 64) {
        return false;
    }
    const unsigned char *image = (const unsigned char *)module->image;
    size_t image_size = module->image_size;
    memset(meta, 0, sizeof(*meta));
    meta->reg_count = 32;
    meta->max_threads_per_block = 1024;

    if (image[0] != 0x7f || image[1] != 'E' || image[2] != 'L' || image[3] != 'F' ||
        image[4] != 2 || image[5] != 1 || rd16le(image + 18) != LANXIN_ELF_EM_CUDA) {
        return false;
    }

    rm_u64 shoff = rd64le(image + 40);
    rm_u32 eflags = rd32le(image + 48);
    uint16_t shentsize = rd16le(image + 58);
    uint16_t shnum = rd16le(image + 60);
    uint16_t shstrndx = rd16le(image + 62);
    if (shentsize < 64 || shnum == 0 || shnum > 8192 || shstrndx >= shnum ||
        !range_in_image(shoff, (rm_u64)shentsize * shnum, image_size)) {
        return false;
    }

    const unsigned char *shbase = image + shoff;
    const unsigned char *shstr_hdr = shbase + (rm_u64)shstrndx * shentsize;
    rm_u64 shstr_off = rd64le(shstr_hdr + 24);
    rm_u64 shstr_size = rd64le(shstr_hdr + 32);
    if (!range_in_image(shstr_off, shstr_size, image_size)) {
        return false;
    }
    const unsigned char *shstr = image + shstr_off;

    struct elf64_section_view text = {0};
    struct elf64_section_view info = {0};
    struct elf64_section_view const0 = {0};
    bool have_text = false;
    bool have_info = false;
    bool have_const0 = false;

    for (uint16_t i = 0; i < shnum; i++) {
        struct elf64_section_view sec;
        const unsigned char *hdr = shbase + (rm_u64)i * shentsize;
        if (!elf_section_from_header(image, image_size, hdr, i, shstr, shstr_size, &sec) ||
            sec.name == NULL) {
            continue;
        }
        if (!have_text && kernel_text_name_matches(sec.name, kernel_name) &&
            sec.type == LANXIN_ELF_SHT_PROGBITS &&
            (sec.flags & LANXIN_ELF_SHF_EXECINSTR) != 0) {
            text = sec;
            have_text = true;
        } else if (!have_info &&
                   kernel_named_section_matches(sec.name, ".nv.info.", kernel_name)) {
            info = sec;
            have_info = true;
        } else if (!have_const0 &&
                   kernel_named_section_matches(sec.name, ".nv.constant0.", kernel_name)) {
            const0 = sec;
            have_const0 = true;
        }
    }

    for (uint16_t i = 0; i < shnum; i++) {
        struct elf64_section_view symtab;
        const unsigned char *hdr = shbase + (rm_u64)i * shentsize;
        if (!elf_section_from_header(image, image_size, hdr, i, shstr, shstr_size, &symtab) ||
            symtab.type != LANXIN_ELF_SHT_SYMTAB || symtab.entsize < 24 ||
            symtab.link >= shnum || symtab.size == 0) {
            continue;
        }
        struct elf64_section_view strtab;
        const unsigned char *str_hdr = shbase + (rm_u64)symtab.link * shentsize;
        if (!elf_section_from_header(image, image_size, str_hdr, symtab.link,
                                     shstr, shstr_size, &strtab) ||
            strtab.type != LANXIN_ELF_SHT_STRTAB) {
            continue;
        }
        const unsigned char *strbase = image + strtab.offset;
        rm_u64 sym_count = symtab.size / symtab.entsize;
        for (rm_u64 j = 0; j < sym_count; j++) {
            const unsigned char *sym = image + symtab.offset + j * symtab.entsize;
            const char *sym_name = elf_string_at(strbase, strtab.size, rd32le(sym + 0));
            rm_u32 sym_type = sym[4] & 0xfu;
            uint16_t shndx = rd16le(sym + 6);
            if (sym_name == NULL || strcmp(sym_name, kernel_name) != 0 ||
                (sym_type != LANXIN_ELF_STT_FUNC && sym_type != LANXIN_ELF_STT_CUDA_FUNC) ||
                shndx >= shnum) {
                continue;
            }
            struct elf64_section_view sym_sec;
            const unsigned char *sym_shdr = shbase + (rm_u64)shndx * shentsize;
            if (elf_section_from_header(image, image_size, sym_shdr, shndx,
                                        shstr, shstr_size, &sym_sec) &&
                sym_sec.type == LANXIN_ELF_SHT_PROGBITS &&
                (sym_sec.flags & LANXIN_ELF_SHF_EXECINSTR) != 0) {
                text = sym_sec;
                have_text = true;
                meta->symbol_value = rd64le(sym + 8);
                meta->symbol_size = rd64le(sym + 16);
            }
        }
    }

    if (!have_text) {
        return false;
    }
    meta->valid = true;
    meta->text_file_offset = text.offset;
    meta->text_size = meta->symbol_size != 0 ? meta->symbol_size : text.size;
    meta->text_addr = text.addr;
    meta->sm_version = eflags & 0xffu;
    if (meta->sm_version == 0 && eflags != 0) {
        meta->sm_version = eflags;
    }
    if (meta->symbol_value != 0 && meta->symbol_value >= text.addr &&
        meta->symbol_value < text.addr + text.size) {
        meta->qmd_program_offset = (rm_u32)(text.offset + (meta->symbol_value - text.addr));
    } else if (meta->symbol_value != 0 && meta->symbol_value < text.size) {
        meta->qmd_program_offset = (rm_u32)(text.offset + meta->symbol_value);
    } else {
        meta->qmd_program_offset = (rm_u32)text.offset;
    }
    if (have_info && info.type == LANXIN_ELF_SHT_CUDA_INFO &&
        range_in_image(info.offset, info.size, image_size)) {
        parse_kernel_nv_info(image + info.offset, info.size, meta);
    }
    if (have_const0 && range_in_image(const0.offset, const0.size, image_size)) {
        meta->const0_file_offset = const0.offset;
        meta->const0_size = const0.size;
        meta->const_mem_bytes = const0.size > 0xffffffffULL ? 0xffffffffu : (rm_u32)const0.size;
    }
    return true;
}

static CUfunction module_find_function(CUmodule module, const char *name)
{
    if (!valid_module(module) || name == NULL) {
        return NULL;
    }
    for (struct CUfunc_st *fn = module->functions; fn != NULL; fn = fn->next) {
        if (strcmp(fn->name, name) == 0) {
            return fn;
        }
    }
    return NULL;
}

static CUresult module_get_or_create_function(CUfunction *hfunc, CUmodule hmod, const char *name)
{
    if (hfunc == NULL || !valid_module(hmod) || name == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUfunction existing = module_find_function(hmod, name);
    if (existing != NULL) {
        *hfunc = existing;
        return CUDA_SUCCESS;
    }
    CUfunction fn = calloc(1, sizeof(*fn));
    if (fn == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    fn->magic = HANDLE_MAGIC_FUNC;
    fn->name = strdup(name);
    if (fn->name == NULL) {
        free(fn);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    fn->name_hash = fnv1a64_str(name);
    fn->module = hmod;
    if (module_find_kernel_metadata(hmod, name, &fn->metadata)) {
        tracef("cuModuleGetFunction cubin name=%s text_off=0x%llx text_size=%llu program_offset=0x%x regs=%u static_smem=%u local=%u const0=0x%llx/%llu max_threads=%u sm=%u",
               name,
               (unsigned long long)fn->metadata.text_file_offset,
               (unsigned long long)fn->metadata.text_size,
               fn->metadata.qmd_program_offset,
               fn->metadata.reg_count,
               fn->metadata.static_shared_mem_bytes,
               fn->metadata.local_mem_bytes,
               (unsigned long long)fn->metadata.const0_file_offset,
               (unsigned long long)fn->metadata.const0_size,
               fn->metadata.max_threads_per_block,
               fn->metadata.sm_version);
    }
    fn->next = hmod->functions;
    hmod->functions = fn;
    *hfunc = fn;
    return CUDA_SUCCESS;
}

static CUresult module_discover_cubin_kernels(CUmodule module)
{
    if (!valid_module(module)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    if (module->kernels_discovered) {
        return CUDA_SUCCESS;
    }
    if (module->image == NULL || module->image_size < 64) {
        module->kernels_discovered = true;
        return CUDA_SUCCESS;
    }

    const unsigned char *image = (const unsigned char *)module->image;
    size_t image_size = module->image_size;
    if (image[0] != 0x7f || image[1] != 'E' || image[2] != 'L' || image[3] != 'F' ||
        image[4] != 2 || image[5] != 1 || rd16le(image + 18) != LANXIN_ELF_EM_CUDA) {
        module->kernels_discovered = true;
        return CUDA_SUCCESS;
    }

    rm_u64 shoff = rd64le(image + 40);
    uint16_t shentsize = rd16le(image + 58);
    uint16_t shnum = rd16le(image + 60);
    uint16_t shstrndx = rd16le(image + 62);
    if (shentsize < 64 || shnum == 0 || shnum > 8192 || shstrndx >= shnum ||
        !range_in_image(shoff, (rm_u64)shentsize * shnum, image_size)) {
        module->kernels_discovered = true;
        return CUDA_SUCCESS;
    }

    const unsigned char *shbase = image + shoff;
    const unsigned char *shstr_hdr = shbase + (rm_u64)shstrndx * shentsize;
    rm_u64 shstr_off = rd64le(shstr_hdr + 24);
    rm_u64 shstr_size = rd64le(shstr_hdr + 32);
    if (!range_in_image(shstr_off, shstr_size, image_size)) {
        module->kernels_discovered = true;
        return CUDA_SUCCESS;
    }
    const unsigned char *shstr = image + shstr_off;

    unsigned int discovered = 0;
    for (uint16_t i = 0; i < shnum; i++) {
        struct elf64_section_view sec;
        const unsigned char *hdr = shbase + (rm_u64)i * shentsize;
        if (!elf_section_from_header(image, image_size, hdr, i, shstr, shstr_size, &sec) ||
            sec.name == NULL || sec.type != LANXIN_ELF_SHT_PROGBITS ||
            (sec.flags & LANXIN_ELF_SHF_EXECINSTR) == 0) {
            continue;
        }
        const char *kernel_name = kernel_name_from_text_section(sec.name);
        if (kernel_name == NULL || module_find_function(module, kernel_name) != NULL) {
            continue;
        }
        CUfunction fn = NULL;
        CUresult result = module_get_or_create_function(&fn, module, kernel_name);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        discovered++;
    }

    module->kernels_discovered = true;
    if (discovered != 0) {
        tracef("cuModuleDiscoverKernels module=%s count=%u",
               module->name != NULL ? module->name : "<module>", discovered);
    }
    return CUDA_SUCCESS;
}

static rm_u64 now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (rm_u64)ts.tv_sec * 1000000000ULL + (rm_u64)ts.tv_nsec;
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

static int rm_map_cpu_with_parent_flags_locked(int map_fd, rm_handle hDevice, rm_handle hMemory,
                                               size_t size, rm_u32 flags,
                                               rm_p64 *linear, void **cpu,
                                               const char *label)
{
    rm_nvos33_with_fd_t api;
    memset(&api, 0, sizeof(api));
    api.params.hClient = g.rm_client;
    api.params.hDevice = hDevice != 0 ? hDevice : g.rm_device;
    api.params.hMemory = hMemory;
    api.params.length = size;
    api.params.flags = flags;
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
        unmap_api.hDevice = api.params.hDevice;
        unmap_api.hMemory = hMemory;
        unmap_api.pLinearAddress = api.params.pLinearAddress;
        (void)rm_ioctl_xfer(g.ctl_fd, RM_ESC_RM_UNMAP_MEMORY, &unmap_api, sizeof(unmap_api));
        return -1;
    }
    *linear = api.params.pLinearAddress;
    *cpu = addr;
    return 0;
}

static int rm_map_cpu_with_parent_locked(int map_fd, rm_handle hDevice, rm_handle hMemory,
                                         size_t size, rm_p64 *linear, void **cpu,
                                         const char *label)
{
    return rm_map_cpu_with_parent_flags_locked(map_fd, hDevice, hMemory, size, 0,
                                               linear, cpu, label);
}

static int rm_map_cpu_locked(int map_fd, rm_handle hMemory, size_t size,
                             rm_p64 *linear, void **cpu, const char *label)
{
    return rm_map_cpu_with_parent_locked(map_fd, g.rm_device, hMemory, size, linear, cpu, label);
}

static void rm_unmap_cpu_with_parent_locked(rm_handle hDevice, rm_handle hMemory,
                                            rm_p64 linear, void *cpu, size_t size)
{
    if (cpu != NULL && size != 0) {
        munmap(cpu, size);
    }
    if (hMemory != 0 && linear != 0) {
        rm_nvos34_t api;
        memset(&api, 0, sizeof(api));
        api.hClient = g.rm_client;
        api.hDevice = hDevice != 0 ? hDevice : g.rm_device;
        api.hMemory = hMemory;
        api.pLinearAddress = linear;
        int rc = rm_ioctl_xfer(g.ctl_fd, RM_ESC_RM_UNMAP_MEMORY, &api, sizeof(api));
        tracef("RM unmap cpu memory=0x%x ioctl=%d errno=%d status=0x%08x",
               hMemory, rc, rc == 0 ? 0 : errno, api.status);
    }
}

static void rm_unmap_cpu_locked(rm_handle hMemory, rm_p64 linear, void *cpu, size_t size)
{
    rm_unmap_cpu_with_parent_locked(g.rm_device, hMemory, linear, cpu, size);
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
    size_t gpfifo_size = g.rm_gpfifo_size != 0 ? g.rm_gpfifo_size : page;
    if (g.rm_compute != 0) {
        rm_free_object_locked(g.rm_client, g.rm_channel, g.rm_compute);
    }
    if (g.rm_channel != 0) {
        rm_free_object_locked(g.rm_client, g.rm_device, g.rm_channel);
    }
    rm_unmap_cpu_with_parent_locked(g.rm_usermode_parent, g.rm_usermode,
                                    g.rm_usermode_linear, g.rm_usermode_cpu,
                                    RM_NVC361_NV_USERMODE_SIZE);
    if (g.rm_usermode != 0) {
        rm_free_object_locked(g.rm_client,
                              g.rm_usermode_parent != 0 ? g.rm_usermode_parent : g.rm_device,
                              g.rm_usermode);
    }
    rm_unmap_cpu_locked(g.rm_userd, g.rm_userd_linear, g.rm_userd_cpu, page);
    if (g.rm_userd != 0) {
        rm_free_object_locked(g.rm_client, g.rm_device, g.rm_userd);
    }
    rm_unmap_dma_locked(g.rm_vaspace, g.rm_gpfifo, g.rm_gpfifo_va, gpfifo_size);
    rm_unmap_cpu_locked(g.rm_gpfifo, g.rm_gpfifo_linear, g.rm_gpfifo_cpu, gpfifo_size);
    if (g.rm_gpfifo != 0) {
        rm_free_object_locked(g.rm_client, g.rm_device, g.rm_gpfifo);
    }
    if (g.rm_error_ctxdma != 0) {
        rm_free_object_locked(g.rm_client, g.rm_device, g.rm_error_ctxdma);
    }
    rm_unmap_cpu_locked(g.rm_notifier, g.rm_notifier_linear, g.rm_notifier_cpu, page);
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
    if (g.rm_usermode_fd >= 0) {
        close(g.rm_usermode_fd);
    }
    if (g.rm_notifier_fd >= 0) {
        close(g.rm_notifier_fd);
    }
    g.rm_notifier_fd = -1;
    g.rm_gpfifo_fd = -1;
    g.rm_userd_fd = -1;
    g.rm_usermode_fd = -1;
    g.rm_vaspace = 0;
    g.rm_notifier = 0;
    g.rm_error_ctxdma = 0;
    g.rm_gpfifo = 0;
    g.rm_userd = 0;
    g.rm_usermode = 0;
    g.rm_usermode_parent = 0;
    g.rm_channel = 0;
    g.rm_compute = 0;
    g.rm_notifier_va = 0;
    g.rm_notifier_linear = 0;
    g.rm_gpfifo_va = 0;
    g.rm_gpfifo_linear = 0;
    g.rm_userd_linear = 0;
    g.rm_usermode_linear = 0;
    g.rm_gpfifo_size = 0;
    g.rm_notifier_cpu = NULL;
    g.rm_gpfifo_cpu = NULL;
    g.rm_userd_cpu = NULL;
    g.rm_usermode_cpu = NULL;
    g.rm_work_submit_token = 0;
    g.rm_doorbell_token = 0;
    g.rm_token_notifier_index = 0;
    g.rm_gpfifo_put = 0;
    g.rm_usermode_class = 0;
    g.rm_compute_class = 0;
    g.rm_compute_class_engine_id = 0;
    g.rm_channel_ready = false;
}

static int rm_alloc_usermode_locked(void)
{
    if (env_disabled("LANXIN_NVIDIA_CUDA_DOORBELL")) {
        tracef("RM usermode doorbell disabled by LANXIN_NVIDIA_CUDA_DOORBELL=0");
        return 0;
    }

    static const rm_u32 usermode_classes[] = {
        RM_BLACKWELL_USERMODE_A,
        RM_HOPPER_USERMODE_A,
        RM_VOLTA_USERMODE_A,
    };
    rm_handle parents[2] = {
        g.rm_subdevice != 0 ? g.rm_subdevice : g.rm_device,
        g.rm_device,
    };

    for (size_t p = 0; p < sizeof(parents) / sizeof(parents[0]); p++) {
        rm_handle parent = parents[p];
        if (parent == 0 || (p != 0 && parent == parents[0])) {
            continue;
        }
        for (size_t i = 0; i < sizeof(usermode_classes) / sizeof(usermode_classes[0]); i++) {
            rm_hopper_usermode_params_t hopper_params = {.bBar1Mapping = 1, .bPriv = 0};
            void *params = usermode_classes[i] == RM_VOLTA_USERMODE_A ? NULL : &hopper_params;
            rm_u32 params_size = params != NULL ? (rm_u32)sizeof(hopper_params) : 0;
            rm_handle usermode = g.next_rm_handle++;
            if (rm_alloc_object_locked(g.rm_client, parent, &usermode, usermode_classes[i],
                                       params, params_size, "rm_alloc_usermode") != 0) {
                continue;
            }
            if (rm_map_cpu_with_parent_flags_locked(g.rm_usermode_fd, parent, usermode,
                                                    RM_NVC361_NV_USERMODE_SIZE,
                                                    RM_NVOS33_FLAGS_ACCESS_WRITE_ONLY,
                                                    &g.rm_usermode_linear, &g.rm_usermode_cpu,
                                                    "rm_map_usermode_cpu") != 0) {
                rm_free_object_locked(g.rm_client, parent, usermode);
                continue;
            }
            g.rm_usermode = usermode;
            g.rm_usermode_parent = parent;
            g.rm_usermode_class = usermode_classes[i];
            tracef("RM usermode doorbell ready object=0x%x parent=0x%x class=0x%x cpu=%p",
                   g.rm_usermode, g.rm_usermode_parent, g.rm_usermode_class,
                   g.rm_usermode_cpu);
            return 0;
        }
    }

    tracef("RM usermode doorbell allocation failed");
    return -1;
}

static rm_u32 rm_read_notifier_info32_locked(rm_u32 index)
{
    if (g.rm_notifier_cpu == NULL) {
        return 0;
    }
    size_t offset = (size_t)index * RM_NV_NOTIFICATION_BYTES + RM_NV_NOTIFICATION_INFO32_OFFSET;
    if (offset + sizeof(rm_u32) > page_align_size(4096)) {
        return 0;
    }
    volatile rm_u32 *info32 = (volatile rm_u32 *)((volatile uint8_t *)g.rm_notifier_cpu + offset);
    __sync_synchronize();
    return *info32;
}

static void rm_channel_kickoff_locked(rm_u32 old_put, rm_u32 new_put)
{
    volatile rm_u32 *control = (volatile rm_u32 *)g.rm_userd_cpu;
    if (control != NULL) {
        control[0x8c / 4] = new_put;
    }
    __sync_synchronize();

    if (!env_disabled("LANXIN_NVIDIA_CUDA_DOORBELL") &&
        g.rm_usermode_cpu != NULL && g.rm_doorbell_token != 0) {
        volatile rm_u32 *doorbell =
            (volatile rm_u32 *)((volatile uint8_t *)g.rm_usermode_cpu +
                                RM_NVC361_NOTIFY_CHANNEL_PENDING);
        *doorbell = g.rm_doorbell_token;
        __sync_synchronize();
        tracef("RM doorbell kickoff old_put=%u new_put=%u token=0x%08x usermode=0x%x class=0x%x",
               old_put, new_put, g.rm_doorbell_token, g.rm_usermode, g.rm_usermode_class);
    } else {
        tracef("RM UserD-only kickoff old_put=%u new_put=%u token=0x%08x usermode=%p",
               old_put, new_put, g.rm_doorbell_token, g.rm_usermode_cpu);
    }
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
    size_t gpfifo_bytes = page_align_size((size_t)env_ull("LANXIN_NVIDIA_CUDA_GPFIFO_BYTES", 65536));
    if (gpfifo_bytes < page) {
        gpfifo_bytes = page;
    }
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

    g.rm_notifier_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    g.rm_gpfifo_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    g.rm_userd_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    g.rm_usermode_fd = g.gpu_fd >= 0 ? dup(g.gpu_fd) : -1;
    if (g.rm_usermode_fd < 0) {
        g.rm_usermode_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    }
    if (g.rm_notifier_fd < 0 || g.rm_gpfifo_fd < 0 ||
        g.rm_userd_fd < 0 || g.rm_usermode_fd < 0) {
        tracef("RM channel fd open failed notifier=%d gpfifo=%d userd=%d usermode=%d errno=%d",
               g.rm_notifier_fd, g.rm_gpfifo_fd, g.rm_userd_fd, g.rm_usermode_fd, errno);
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
    if (rm_alloc_memory_locked(g.gpu_fd, g.rm_notifier_fd, &g.rm_notifier,
                               RM_NV01_MEMORY_SYSTEM, coherent_flags, page,
                               "rm_alloc_notifier") != 0 ||
        rm_map_cpu_locked(g.rm_notifier_fd, g.rm_notifier, page,
                          &g.rm_notifier_linear, &g.rm_notifier_cpu,
                          "rm_map_notifier_cpu") != 0 ||
        rm_map_dma_locked(g.rm_vaspace, g.rm_notifier, page, &g.rm_notifier_va,
                          "rm_map_notifier_dma") != 0) {
        goto fail;
    }
    memset(g.rm_notifier_cpu, 0, page);

    g.rm_error_ctxdma = g.next_rm_handle++;
    if (rm_alloc_ctxdma_locked(g.rm_notifier, 0xff, &g.rm_error_ctxdma) != 0) {
        goto fail;
    }

    g.rm_gpfifo = g.next_rm_handle++;
    if (rm_alloc_memory_locked(g.gpu_fd, g.rm_gpfifo_fd, &g.rm_gpfifo,
                               RM_NV01_MEMORY_SYSTEM, sys_flags, gpfifo_bytes,
                               "rm_alloc_gpfifo") != 0 ||
        rm_map_cpu_locked(g.rm_gpfifo_fd, g.rm_gpfifo, gpfifo_bytes, &g.rm_gpfifo_linear,
                          &g.rm_gpfifo_cpu, "rm_map_gpfifo_cpu") != 0 ||
        rm_map_dma_locked(g.rm_vaspace, g.rm_gpfifo, gpfifo_bytes, &g.rm_gpfifo_va,
                          "rm_map_gpfifo_dma") != 0) {
        goto fail;
    }
    g.rm_gpfifo_size = gpfifo_bytes;
    memset(g.rm_gpfifo_cpu, 0, gpfifo_bytes);

    g.rm_userd = g.next_rm_handle++;
    if (rm_alloc_memory_locked(g.gpu_fd, g.rm_userd_fd, &g.rm_userd,
                               RM_NV01_MEMORY_SYSTEM, sys_flags, page,
                               "rm_alloc_userd") != 0 ||
        rm_map_cpu_locked(g.rm_userd_fd, g.rm_userd, page, &g.rm_userd_linear,
                          &g.rm_userd_cpu, "rm_map_userd_cpu") != 0) {
        goto fail;
    }
    memset(g.rm_userd_cpu, 0, page);

    if (rm_alloc_usermode_locked() != 0) {
        goto fail;
    }

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
    rm_channel_token_notif_index_params_t notif = {
        .index = RM_NV_CHANNELGPFIFO_NOTIFICATION_TYPE_SIZE_1,
    };
    rm_channel_token_params_t token;
    memset(&token, 0, sizeof(token));
    if (rm_control_locked(g.rm_channel, RM_NVA06F_CTRL_CMD_BIND,
                          &bind, sizeof(bind), "rm_channel_bind") != 0 ||
        rm_control_locked(g.rm_channel, RM_NVA06F_CTRL_CMD_GPFIFO_SCHEDULE,
                          &schedule, sizeof(schedule), "rm_channel_schedule") != 0 ||
        (!env_disabled("LANXIN_NVIDIA_CUDA_DOORBELL") &&
         rm_control_locked(g.rm_channel, RM_NVC36F_CTRL_CMD_GPFIFO_SET_WORK_SUBMIT_TOKEN_NOTIF_INDEX,
                           &notif, sizeof(notif), "rm_channel_token_notif_index") != 0) ||
        rm_control_locked(g.rm_channel, RM_NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
                          &token, sizeof(token), "rm_channel_token") != 0) {
        goto fail;
    }
    g.rm_work_submit_token = token.workSubmitToken;
    g.rm_token_notifier_index = notif.index;
    g.rm_doorbell_token = rm_read_notifier_info32_locked(g.rm_token_notifier_index);
    if (g.rm_doorbell_token == 0) {
        g.rm_doorbell_token = g.rm_work_submit_token;
    }
    if (!env_disabled("LANXIN_NVIDIA_CUDA_DOORBELL") && g.rm_doorbell_token == 0) {
        tracef("RM doorbell token is zero");
        goto fail;
    }
    g.rm_channel_ready = true;
    tracef("RM channel ready channel=0x%x vaspace=0x%x gpfifo=0x%x/0x%llx userd=0x%x usermode=0x%x notifier_index=%u token=0x%08x doorbell=0x%08x",
           g.rm_channel, g.rm_vaspace, g.rm_gpfifo,
           (unsigned long long)g.rm_gpfifo_va, g.rm_userd, g.rm_usermode,
           g.rm_token_notifier_index, g.rm_work_submit_token, g.rm_doorbell_token);
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

static void rm_qmd_set_bits(rm_u32 *words, unsigned int lo, unsigned int hi, rm_u64 value)
{
    if (words == NULL || hi < lo || hi >= LANXIN_HW_QMD_WORDS * 32U) {
        return;
    }
    for (unsigned int bit = lo; bit <= hi; bit++) {
        unsigned int src = bit - lo;
        unsigned int word = bit / 32U;
        unsigned int shift = bit % 32U;
        rm_u32 mask = 1u << shift;
        if ((value >> src) & 1ULL) {
            words[word] |= mask;
        } else {
            words[word] &= ~mask;
        }
    }
}

static rm_u32 rm_qmd_u32_clamp(rm_u32 value, rm_u32 fallback)
{
    return value == 0 ? fallback : value;
}

static rm_u32 rm_qmd_shared_mem_units(rm_u32 bytes)
{
    rm_u64 units = ((rm_u64)bytes + 255ULL) / 256ULL;
    return units > 0x3ffffULL ? 0x3ffffu : (rm_u32)units;
}

static rm_u64 rm_completion_host_va(const struct launch_staging *launch)
{
    return launch->completion.gpu_va + offsetof(struct lanxin_launch_completion, host_progress);
}

static rm_u64 rm_completion_qmd_va(const struct launch_staging *launch)
{
    return launch->completion.gpu_va + offsetof(struct lanxin_launch_completion, qmd_done);
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

static void rm_stage_release_locked(struct rm_stage_buffer *buf)
{
    if (buf == NULL || buf->memory == 0) {
        return;
    }
    rm_unmap_dma_locked(g.rm_vaspace, buf->memory, buf->gpu_va, buf->size);
    rm_unmap_cpu_locked(buf->memory, buf->linear, buf->cpu, buf->size);
    rm_free_object_locked(g.rm_client, g.rm_device, buf->memory);
    if (buf->map_fd > 0) {
        close(buf->map_fd);
    }
    memset(buf, 0, sizeof(*buf));
}

static int rm_stage_alloc_locked(struct rm_stage_buffer *buf, size_t size, const char *label)
{
    if (buf == NULL) {
        return -1;
    }
    size_t mapped_size = page_align_size(size);
    if (buf->memory != 0 && buf->cpu != NULL && buf->gpu_va != 0 && buf->size >= mapped_size) {
        memset(buf->cpu, 0, buf->size);
        return 0;
    }
    rm_stage_release_locked(buf);
    if (rm_channel_init_locked() != 0) {
        return -1;
    }
    int map_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    if (map_fd < 0) {
        tracef("%s map fd open failed errno=%d", label, errno);
        return -1;
    }

    rm_u32 flags = RM_NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS |
                   RM_NVOS02_FLAGS_LOCATION_PCI |
                   RM_NVOS02_FLAGS_COHERENCY_WRITE_COMBINE |
                   RM_NVOS02_FLAGS_GPU_CACHEABLE_YES |
                   RM_NVOS02_FLAGS_MAPPING_NO_MAP;
    buf->memory = g.next_rm_handle++;
    buf->size = mapped_size;
    buf->map_fd = map_fd;
    if (rm_alloc_memory_locked(g.gpu_fd, buf->map_fd, &buf->memory,
                               RM_NV01_MEMORY_SYSTEM, flags, mapped_size, label) != 0 ||
        rm_map_cpu_locked(buf->map_fd, buf->memory, mapped_size,
                          &buf->linear, &buf->cpu, label) != 0 ||
        rm_map_dma_locked(g.rm_vaspace, buf->memory, mapped_size,
                          &buf->gpu_va, label) != 0) {
        rm_stage_release_locked(buf);
        return -1;
    }
    memset(buf->cpu, 0, mapped_size);
    tracef("%s staged handle=0x%x cpu=%p gpu_va=0x%llx bytes=%zu",
           label, buf->memory, buf->cpu, (unsigned long long)buf->gpu_va, mapped_size);
    return 0;
}

static size_t launch_code_stage_limit(void)
{
    unsigned long long limit = env_ull("LANXIN_NVIDIA_CUDA_CODE_STAGE_MAX_BYTES", 16ULL * 1024ULL * 1024ULL);
    if (limit == 0) {
        limit = 1;
    }
    return (size_t)limit;
}

static int rm_stage_module_code_locked(CUfunction f)
{
    if (!valid_function(f) || !valid_module(f->module)) {
        return -1;
    }
    CUmodule module = f->module;
    if (module->image_hash == 0 && module->image != NULL && module->image_size != 0) {
        module->image_hash = fnv1a64_bytes(module->image, module->image_size);
    } else if (module->image_hash == 0) {
        module->image_hash = fnv1a64_str(module->name);
    }

    rm_u32 layout = LANXIN_CODE_STAGE_WHOLE_IMAGE;
    const unsigned char *copy_src = module->image != NULL ?
                                    (const unsigned char *)module->image : NULL;
    size_t copy_bytes = module->image != NULL ? module->image_size : 0;
    rm_u64 text_file_offset = 0;
    rm_u64 text_size = 0;
    if (env_enabled("LANXIN_NVIDIA_CUDA_CODE_STAGE_TEXT") &&
        f->metadata.valid && module->image != NULL &&
        range_in_image(f->metadata.text_file_offset, f->metadata.text_size,
                       module->image_size) &&
        f->metadata.text_size != 0) {
        layout = LANXIN_CODE_STAGE_TEXT_ONLY;
        text_file_offset = f->metadata.text_file_offset;
        text_size = f->metadata.text_size;
        copy_src = (const unsigned char *)module->image + f->metadata.text_file_offset;
        copy_bytes = f->metadata.text_size > (rm_u64)SIZE_MAX ?
                     SIZE_MAX : (size_t)f->metadata.text_size;
    }
    size_t limit = launch_code_stage_limit();
    if (copy_bytes > limit) {
        copy_bytes = limit;
    }
    if (copy_bytes == 0) {
        copy_bytes = 64;
    }
    if (rm_stage_alloc_locked(&module->code, copy_bytes, "rm_stage_code_object") != 0) {
        return -1;
    }
    if (copy_src != NULL && copy_bytes != 0) {
        memcpy(module->code.cpu, copy_src, copy_bytes);
    } else {
        snprintf((char *)module->code.cpu, module->code.size, "lanxin fake cuda code object: %s",
                 module->name != NULL ? module->name : "<module>");
    }
    module->staged_code_bytes = copy_bytes;
    module->staged_code_layout = layout;
    module->staged_text_file_offset = text_file_offset;
    module->staged_text_size = text_size;
    module->staged_function_name_hash = f->name_hash;
    __sync_synchronize();
    tracef("RM staged code object module=%s image_bytes=%zu staged=%zu layout=%s text_off=0x%llx text_size=%llu hash=0x%016llx gpu_va=0x%llx",
           module->name != NULL ? module->name : "<module>", module->image_size, module->staged_code_bytes,
           layout == LANXIN_CODE_STAGE_TEXT_ONLY ? "text" : "whole",
           (unsigned long long)text_file_offset, (unsigned long long)text_size,
           (unsigned long long)module->image_hash, (unsigned long long)module->code.gpu_va);
    return 0;
}

static void rm_build_hardware_qmd_locked(CUfunction f, const struct launch_request *req,
                                         struct launch_staging *launch,
                                         struct lanxin_launch_completion *completion)
{
    rm_u32 *qmd = (rm_u32 *)launch->qmd.cpu;
    struct lanxin_qmd_staging_desc *meta =
        (struct lanxin_qmd_staging_desc *)((uint8_t *)launch->qmd.cpu + LANXIN_HW_QMD_BYTES);
    memset(qmd, 0, LANXIN_HW_QMD_BYTES);
    memset(meta, 0, sizeof(*meta));

    rm_u32 grid_x = rm_qmd_u32_clamp(req->grid[0], 1);
    rm_u32 grid_y = rm_qmd_u32_clamp(req->grid[1], 1);
    rm_u32 grid_z = rm_qmd_u32_clamp(req->grid[2], 1);
    rm_u32 block_x = rm_qmd_u32_clamp(req->block[0], 1);
    rm_u32 block_y = rm_qmd_u32_clamp(req->block[1], 1);
    rm_u32 block_z = rm_qmd_u32_clamp(req->block[2], 1);
    rm_u32 qmd_payload = (rm_u32)(launch->launch_id & 0xffffffffu);
    if (qmd_payload == 0) {
        qmd_payload = 1;
    }

    rm_u64 params_va = launch->params.gpu_va;
    rm_u64 qmd_release_va = rm_completion_qmd_va(launch);
    bool code_text_only = f->metadata.valid &&
                          f->module->staged_code_layout == LANXIN_CODE_STAGE_TEXT_ONLY &&
                          f->module->staged_function_name_hash == f->name_hash &&
                          f->module->staged_text_file_offset == f->metadata.text_file_offset;
    rm_u32 parsed_program_offset = f->metadata.valid ? f->metadata.qmd_program_offset : 0;
    if (code_text_only) {
        parsed_program_offset = 0;
    }
    rm_u32 parsed_reg_count = f->metadata.valid && f->metadata.reg_count != 0 ?
                              f->metadata.reg_count : 32;
    rm_u32 parsed_sass_version = f->metadata.valid ? f->metadata.sm_version : 0;
    rm_u32 parsed_static_smem = f->metadata.valid ? f->metadata.static_shared_mem_bytes : 0;
    rm_u32 parsed_local_mem = f->metadata.valid ? f->metadata.local_mem_bytes : 0;
    rm_u32 shared_total = req->shared_mem_bytes + parsed_static_smem;
    if (shared_total < req->shared_mem_bytes) {
        shared_total = UINT32_MAX;
    }
    rm_u32 qmd_version = env_u32("LANXIN_NVIDIA_CUDA_QMD_VERSION", LANXIN_QMD_MINOR_VERSION);
    rm_u32 qmd_major = env_u32("LANXIN_NVIDIA_CUDA_QMD_MAJOR_VERSION", LANXIN_QMD_MAJOR_VERSION);
    rm_u32 program_offset = env_u32("LANXIN_NVIDIA_CUDA_QMD_PROGRAM_OFFSET",
                                    parsed_program_offset);
    rm_u32 reg_count = env_u32("LANXIN_NVIDIA_CUDA_QMD_REGISTER_COUNT", parsed_reg_count);
    rm_u32 sass_version = env_u32("LANXIN_NVIDIA_CUDA_QMD_SASS_VERSION", parsed_sass_version);
    rm_u32 local_mem = env_u32("LANXIN_NVIDIA_CUDA_QMD_LOCAL_MEM_BYTES", parsed_local_mem);

    rm_qmd_set_bits(qmd, 200, 200, 1);                      /* schedule-on-put-update */
    rm_qmd_set_bits(qmd, 202, 202, 1);                      /* release0 enabled */
    rm_qmd_set_bits(qmd, 204, 204,
                    env_disabled("LANXIN_NVIDIA_CUDA_QMD_REQUIRE_PCAS") ? 0 : 1);
    rm_qmd_set_bits(qmd, 250, 255, 0x3f);                   /* invalidate shader/texture caches */
    rm_qmd_set_bits(qmd, 256, 287, program_offset);         /* program offset from SET_PROGRAM_REGION */
    rm_qmd_set_bits(qmd, 366, 366, 1);                      /* FE sysmembar on release */
    rm_qmd_set_bits(qmd, 368, 369, 1);                      /* CWD sysmembar */
    rm_qmd_set_bits(qmd, 378, 378, 1);                      /* no API call-limit check */
    rm_qmd_set_bits(qmd, 384, 415, grid_x);
    rm_qmd_set_bits(qmd, 416, 431, grid_y);
    rm_qmd_set_bits(qmd, 432, 447, grid_z);
    rm_qmd_set_bits(qmd, 448, 479, 0);
    rm_qmd_set_bits(qmd, 480, 495, 0);
    rm_qmd_set_bits(qmd, 496, 511, 0);
    rm_qmd_set_bits(qmd, 544, 561, rm_qmd_shared_mem_units(shared_total));
    rm_qmd_set_bits(qmd, 576, 579, qmd_version & 0xfu);
    rm_qmd_set_bits(qmd, 580, 583, qmd_major & 0xfu);
    rm_qmd_set_bits(qmd, 592, 607, block_x);
    rm_qmd_set_bits(qmd, 608, 623, block_y);
    rm_qmd_set_bits(qmd, 624, 639, block_z);
    if (params_va != 0 && launch->params.size != 0) {
        rm_qmd_set_bits(qmd, 640, 640, 1);                  /* constant buffer 0 valid */
        rm_qmd_set_bits(qmd, 928, 959, (rm_u32)(params_va & 0xffffffffu));
        rm_qmd_set_bits(qmd, 960, 967, (rm_u32)((params_va >> 32) & 0xffu));
        rm_qmd_set_bits(qmd, 974, 974, 1);
        rm_qmd_set_bits(qmd, 975, 991, (rm_u32)launch->params.size);
    }
    rm_qmd_set_bits(qmd, 736, 767, (rm_u32)(qmd_release_va & 0xffffffffu));
    rm_qmd_set_bits(qmd, 768, 775, (rm_u32)((qmd_release_va >> 32) & 0xffu));
    rm_qmd_set_bits(qmd, 794, 794, 0);                      /* no release reduction */
    rm_qmd_set_bits(qmd, 799, 799, 1);                      /* one-word release */
    rm_qmd_set_bits(qmd, 800, 831, qmd_payload);
    rm_qmd_set_bits(qmd, 1440, 1463, local_mem & 0xffffffu);
    rm_qmd_set_bits(qmd, 1496, 1503, reg_count);
    rm_qmd_set_bits(qmd, 1528, 1535, sass_version);

    completion->expected_qmd_done = qmd_payload;

    meta->magic = LANXIN_QMD_STAGING_MAGIC;
    meta->version = LANXIN_QMD_STAGING_VERSION;
    meta->size = sizeof(*meta);
    meta->launch_id = launch->launch_id;
    meta->grid[0] = grid_x;
    meta->grid[1] = grid_y;
    meta->grid[2] = grid_z;
    meta->block[0] = block_x;
    meta->block[1] = block_y;
    meta->block[2] = block_z;
    meta->shared_mem_bytes = shared_total;
    meta->code_va = f->module->code.gpu_va;
    meta->code_bytes = f->module->staged_code_bytes;
    meta->qmd_va = launch->qmd.gpu_va;
    meta->params_va = params_va;
    meta->params_bytes = req->params_size;
    meta->completion_va = launch->completion.gpu_va;
    meta->module_image_bytes = f->module->image_size;
    meta->module_image_hash = f->module->image_hash;
    meta->function_name_hash = f->name_hash != 0 ? f->name_hash : fnv1a64_str(f->name);
    meta->compute_class = g.rm_compute_class;
    meta->class_engine_id = g.rm_compute_class_engine_id;
    meta->param_flags = req->param_flags;
    meta->metadata_flags = (f->metadata.valid ? 1u : 0u) |
                           (code_text_only ? 2u : 0u);
    meta->text_file_offset = f->metadata.text_file_offset;
    meta->text_size = f->metadata.text_size;
    meta->const0_file_offset = f->metadata.const0_file_offset;
    meta->const0_size = f->metadata.const0_size;
    meta->qmd_program_offset = program_offset;
    meta->reg_count = reg_count;
    meta->static_shared_mem_bytes = parsed_static_smem;
    meta->local_mem_bytes = local_mem;
    meta->const_mem_bytes = f->metadata.const_mem_bytes;
    meta->max_threads_per_block = f->metadata.max_threads_per_block;
    meta->sm_version = sass_version;
}

static int rm_prepare_launch_packet_locked(CUfunction f, const struct launch_request *req)
{
    if (!valid_function(f) || req == NULL || rm_stage_module_code_locked(f) != 0) {
        return -1;
    }
    struct launch_staging *launch = &f->launch;
    size_t pushbuffer_bytes = page_align_size(4096);
    size_t qmd_bytes = page_align_size(LANXIN_HW_QMD_BYTES + sizeof(struct lanxin_qmd_staging_desc));
    size_t params_bytes = req->params_size != 0 ? req->params_size : 64;
    size_t completion_bytes = page_align_size(sizeof(struct lanxin_launch_completion));
    if (rm_stage_alloc_locked(&launch->pushbuffer, pushbuffer_bytes, "rm_stage_pushbuffer") != 0 ||
        rm_stage_alloc_locked(&launch->qmd, qmd_bytes, "rm_stage_qmd") != 0 ||
        rm_stage_alloc_locked(&launch->params, params_bytes, "rm_stage_params") != 0 ||
        rm_stage_alloc_locked(&launch->completion, completion_bytes, "rm_stage_completion") != 0) {
        return -1;
    }

    if (req->params != NULL && req->params_size != 0) {
        memcpy(launch->params.cpu, req->params, req->params_size);
    }

    launch->launch_id = g.next_id++;
    struct lanxin_launch_completion *completion = (struct lanxin_launch_completion *)launch->completion.cpu;
    memset(completion, 0, sizeof(*completion));
    completion->magic = LANXIN_COMPLETION_MAGIC;
    completion->launch_id = launch->launch_id;
    completion->status = LANXIN_COMPLETION_PENDING;
    completion->timestamp_ns = now_ns();
    completion->expected_host_progress = (rm_u32)(launch->launch_id & 0xffffffffu);
    if (completion->expected_host_progress == 0) {
        completion->expected_host_progress = 1;
    }

    rm_build_hardware_qmd_locked(f, req, launch, completion);
    __sync_synchronize();

    struct lanxin_qmd_staging_desc *qmd_meta =
        (struct lanxin_qmd_staging_desc *)((uint8_t *)launch->qmd.cpu + LANXIN_HW_QMD_BYTES);
    tracef("RM staged hardware QMD launch_id=%llu fn=%s pb=0x%llx qmd=0x%llx code=0x%llx params=0x%llx/%zu completion=0x%llx host_sem=0x%llx qmd_sem=0x%llx grid=%ux%ux%u block=%ux%ux%u qmd_payload=0x%x meta=0x%x program_offset=0x%x regs=%u smem=%u local=%u sm=%u",
           launch->launch_id, f->name != NULL ? f->name : "<unnamed>",
           (unsigned long long)launch->pushbuffer.gpu_va,
           (unsigned long long)launch->qmd.gpu_va,
           (unsigned long long)f->module->code.gpu_va,
           (unsigned long long)launch->params.gpu_va, req->params_size,
           (unsigned long long)launch->completion.gpu_va,
           (unsigned long long)rm_completion_host_va(launch),
           (unsigned long long)rm_completion_qmd_va(launch),
           req->grid[0], req->grid[1], req->grid[2],
           req->block[0], req->block[1], req->block[2],
           completion->expected_qmd_done,
           qmd_meta->metadata_flags, qmd_meta->qmd_program_offset, qmd_meta->reg_count,
           qmd_meta->shared_mem_bytes, qmd_meta->local_mem_bytes, qmd_meta->sm_version);
    return 0;
}

static int rm_poll_launch_completion_locked(CUfunction f, bool require_qmd_release)
{
    if (!valid_function(f) || f->launch.completion.cpu == NULL) {
        return -1;
    }
    unsigned long long timeout_ms = env_ull("LANXIN_NVIDIA_CUDA_COMPLETION_TIMEOUT_MS", 10);
    struct lanxin_launch_completion *completion = (struct lanxin_launch_completion *)f->launch.completion.cpu;
    rm_u64 start = now_ns();
    while (completion->magic == LANXIN_COMPLETION_MAGIC &&
           completion->launch_id == f->launch.launch_id &&
           completion->status == LANXIN_COMPLETION_PENDING) {
        __sync_synchronize();
        bool host_done = completion->host_progress == completion->expected_host_progress;
        bool qmd_done = !require_qmd_release ||
                        completion->qmd_done == completion->expected_qmd_done;
        if (host_done && qmd_done) {
            completion->status = LANXIN_COMPLETION_DONE;
            break;
        }
        rm_u64 elapsed_ns = now_ns() - start;
        if (elapsed_ns >= timeout_ms * 1000000ULL) {
            completion->status = LANXIN_COMPLETION_TIMEOUT;
            tracef("RM launch completion timeout launch_id=%llu status=%u put=%llu host=0x%llx/0x%llx qmd=0x%x/0x%x require_qmd=%d",
                   f->launch.launch_id, completion->status,
                   (unsigned long long)completion->gpfifo_put,
                   (unsigned long long)completion->host_progress,
                   (unsigned long long)completion->expected_host_progress,
                   completion->qmd_done, completion->expected_qmd_done,
                   require_qmd_release ? 1 : 0);
            return -1;
        }
        usleep(1000);
    }
    tracef("RM launch completion observed launch_id=%llu status=%u put=%llu host=0x%llx/0x%llx qmd=0x%x/0x%x require_qmd=%d",
           f->launch.launch_id, completion->status,
           (unsigned long long)completion->gpfifo_put,
           (unsigned long long)completion->host_progress,
           (unsigned long long)completion->expected_host_progress,
           completion->qmd_done, completion->expected_qmd_done,
           require_qmd_release ? 1 : 0);
    return completion->status == LANXIN_COMPLETION_DONE ? 0 : -1;
}

static void rm_release_function_launch_locked(CUfunction f)
{
    if (f == NULL) {
        return;
    }
    rm_stage_release_locked(&f->launch.pushbuffer);
    rm_stage_release_locked(&f->launch.qmd);
    rm_stage_release_locked(&f->launch.params);
    rm_stage_release_locked(&f->launch.completion);
    f->launch.launch_id = 0;
}

static int rm_submit_noop_locked(void)
{
    if (rm_channel_init_locked() != 0) {
        return -1;
    }
    volatile rm_u32 *gp_words = (volatile rm_u32 *)g.rm_gpfifo_cpu;
    rm_u32 entry = g.rm_gpfifo_put % 32U;
    gp_words[entry * 2U] = 0;
    gp_words[entry * 2U + 1U] = 0;
    gp_words[((entry + 1U) % 32U) * 2U] = 0;
    gp_words[((entry + 1U) % 32U) * 2U + 1U] = 0;
    __sync_synchronize();
    rm_u32 old_put = g.rm_gpfifo_put;
    g.rm_gpfifo_put = (entry + 2U) % 32U;
    rm_channel_kickoff_locked(old_put, g.rm_gpfifo_put);
    tracef("RM submitted GPFIFO NOP channel=0x%x put=%u token=0x%08x",
           g.rm_channel, g.rm_gpfifo_put, g.rm_work_submit_token);
    return 0;
}

static rm_u32 env_u32(const char *name, rm_u32 fallback)
{
    unsigned long long value = env_ull(name, fallback);
    if (value > 0xffffffffULL) {
        return fallback;
    }
    return (rm_u32)value;
}

static int rm_submit_compute_set_object_locked(CUfunction f, const struct launch_request *req, bool qmd_requested)
{
    if (rm_compute_init_locked() != 0) {
        return -1;
    }

    bool qmd_stage = !env_disabled("LANXIN_NVIDIA_CUDA_QMD_STAGE");
    bool qmd_ready = false;
    if (qmd_stage && valid_function(f) && req != NULL) {
        if (rm_prepare_launch_packet_locked(f, req) == 0) {
            qmd_ready = true;
        } else if (qmd_requested) {
            return -1;
        } else {
            tracef("RM QMD staging failed; falling back to SET_OBJECT-only PB");
        }
    }

    volatile rm_u32 *gp_words = (volatile rm_u32 *)g.rm_gpfifo_cpu;
    rm_u32 entry = g.rm_gpfifo_put % 32U;
    if (entry & 1U) {
        entry = (entry + 1U) % 32U;
    }
    rm_u32 progress_entry = (entry + 1U) % 32U;
    rm_u32 pb_stride = 0x400u;
    rm_u32 pb_offset = 0x1000u + (entry * pb_stride);
    rm_u32 progress_offset = 0x1000u + (progress_entry * pb_stride);
    volatile rm_u32 *pb_words = NULL;
    volatile rm_u32 *progress_words = NULL;
    rm_u64 pb_va = 0;
    rm_u64 progress_va = 0;
    if (g.rm_gpfifo_cpu != NULL && g.rm_gpfifo_size >= (size_t)pb_offset + pb_stride) {
        pb_words = (volatile rm_u32 *)((volatile uint8_t *)g.rm_gpfifo_cpu + pb_offset);
        memset((void *)pb_words, 0, pb_stride);
        pb_va = g.rm_gpfifo_va + pb_offset;
    } else if (qmd_ready && f->launch.pushbuffer.cpu != NULL && f->launch.pushbuffer.gpu_va != 0) {
        pb_words = (volatile rm_u32 *)f->launch.pushbuffer.cpu;
        memset((void *)pb_words, 0, f->launch.pushbuffer.size);
        pb_va = f->launch.pushbuffer.gpu_va;
    } else {
        pb_offset = 0x200u + (entry * 0x40u);
        pb_words = (volatile rm_u32 *)((volatile uint8_t *)g.rm_gpfifo_cpu + pb_offset);
        pb_va = g.rm_gpfifo_va + pb_offset;
    }
    if (g.rm_gpfifo_cpu != NULL && g.rm_gpfifo_size >= (size_t)progress_offset + pb_stride) {
        progress_words = (volatile rm_u32 *)((volatile uint8_t *)g.rm_gpfifo_cpu + progress_offset);
        memset((void *)progress_words, 0, pb_stride);
        progress_va = g.rm_gpfifo_va + progress_offset;
    }
    rm_u32 pb_count = 0;
    rm_u32 progress_count = 0;
    struct lanxin_launch_completion *completion =
        qmd_ready ? (struct lanxin_launch_completion *)f->launch.completion.cpu : NULL;

    pb_words[pb_count++] = rm_push_method_header(RM_NVA06F_SUBCHANNEL_COMPUTE, RM_COMPUTE_SET_OBJECT,
                                                 1, RM_NVC46F_DMA_SEC_OP_INC_METHOD);
    pb_words[pb_count++] = g.rm_compute_class_engine_id;

    if (qmd_requested && qmd_ready) {
        rm_u64 qmd_va = f->launch.qmd.gpu_va;
        rm_u64 code_va = f->module->code.gpu_va;
        rm_u32 qmd_version_method =
            env_u32("LANXIN_NVIDIA_CUDA_QMD_METHOD_VERSION",
                    (LANXIN_QMD_VERSION_01_06 << 16) | LANXIN_QMD_VERSION_01_06);
        rm_u32 cwd_slots = env_u32("LANXIN_NVIDIA_CUDA_QMD_CWD_SLOT_COUNT", 32);
        rm_u32 pcas_b = env_u32("LANXIN_NVIDIA_CUDA_QMD_PCAS_B", 0);
        rm_u32 pcas_signal =
            env_u32("LANXIN_NVIDIA_CUDA_QMD_PCAS_SIGNAL",
                    (1u << 0) | (1u << 1)); /* invalidate + schedule */

        pb_words[pb_count++] = rm_push_method_header(RM_NVA06F_SUBCHANNEL_COMPUTE,
                                                     RM_COMPUTE_SET_PROGRAM_REGION_A,
                                                     2, RM_NVC46F_DMA_SEC_OP_INC_METHOD);
        pb_words[pb_count++] = (rm_u32)((code_va >> 32) & 0x1ffffu);
        pb_words[pb_count++] = (rm_u32)(code_va & 0xffffffffu);

        pb_words[pb_count++] = rm_push_method_header(RM_NVA06F_SUBCHANNEL_COMPUTE,
                                                     RM_COMPUTE_SET_QMD_VERSION,
                                                     1, RM_NVC46F_DMA_SEC_OP_INC_METHOD);
        pb_words[pb_count++] = qmd_version_method;

        if (env_enabled("LANXIN_NVIDIA_CUDA_CHECK_QMD_VERSION")) {
            pb_words[pb_count++] = rm_push_method_header(RM_NVA06F_SUBCHANNEL_COMPUTE,
                                                         RM_COMPUTE_CHECK_QMD_VERSION,
                                                         1, RM_NVC46F_DMA_SEC_OP_INC_METHOD);
            pb_words[pb_count++] = qmd_version_method;
        }

        pb_words[pb_count++] = rm_push_method_header(RM_NVA06F_SUBCHANNEL_COMPUTE,
                                                     RM_COMPUTE_SET_CWD_SLOT_COUNT,
                                                     1, RM_NVC46F_DMA_SEC_OP_INC_METHOD);
        pb_words[pb_count++] = cwd_slots & 0xffu;

        pb_words[pb_count++] = rm_push_method_header(RM_NVA06F_SUBCHANNEL_COMPUTE,
                                                     RM_COMPUTE_SEND_PCAS_A,
                                                     3, RM_NVC46F_DMA_SEC_OP_INC_METHOD);
        pb_words[pb_count++] = (rm_u32)((qmd_va >> 8) & 0xffffffffu);
        pb_words[pb_count++] = pcas_b;
        pb_words[pb_count++] = pcas_signal;
        if (completion != NULL) {
            completion->qmd_submitted = 1;
        }
        tracef("RM QMD PCAS submit qmd_va=0x%llx qmd_shifted8=0x%x code_region=0x%llx version=0x%08x pcas_b=0x%08x signal=0x%08x",
               (unsigned long long)qmd_va, (rm_u32)((qmd_va >> 8) & 0xffffffffu),
               (unsigned long long)code_va, qmd_version_method, pcas_b, pcas_signal);
    } else if (qmd_requested) {
        tracef("RM QMD submit requested but hardware QMD staging is not ready");
    }

    pb_words[pb_count++] = rm_push_method_header(RM_NVA06F_SUBCHANNEL_COMPUTE, RM_COMPUTE_NO_OPERATION,
                                                 1, RM_NVC46F_DMA_SEC_OP_INC_METHOD);
    pb_words[pb_count++] = 0;
    pb_words[pb_count++] = rm_push_method_header(RM_NVA06F_SUBCHANNEL_COMPUTE, RM_COMPUTE_PIPE_NOP,
                                                 1, RM_NVC46F_DMA_SEC_OP_INC_METHOD);
    pb_words[pb_count++] = 0;

    if (completion != NULL) {
        if (progress_words == NULL || progress_va == 0) {
            tracef("RM progress tracker PB unavailable for completion");
            return -1;
        }
        bool progress_wfi = env_enabled("LANXIN_NVIDIA_CUDA_PROGRESS_WFI");
        rm_u64 host_sem_va = rm_completion_host_va(&f->launch);
        rm_u32 host_payload = (rm_u32)completion->expected_host_progress;
        rm_u32 sem_execute = RM_CHANNEL_SEM_EXECUTE_RELEASE |
                             RM_CHANNEL_SEM_EXECUTE_PAYLOAD_SIZE_32BIT |
                             RM_CHANNEL_SEM_EXECUTE_RELEASE_TIMESTAMP_DIS |
                             (progress_wfi ? RM_CHANNEL_SEM_EXECUTE_RELEASE_WFI_EN : 0);

        progress_words[progress_count++] = rm_push_method_header(RM_CHANNEL_SUBCHANNEL_HOST,
                                                                 RM_CHANNEL_SEM_ADDR_LO,
                                                                 5, RM_NVC46F_DMA_SEC_OP_INC_METHOD);
        progress_words[progress_count++] = (rm_u32)(host_sem_va & 0xffffffffu);
        progress_words[progress_count++] = (rm_u32)((host_sem_va >> 32) & 0xffffffffu);
        progress_words[progress_count++] = host_payload;
        progress_words[progress_count++] = 0;
        progress_words[progress_count++] = sem_execute;
        tracef("RM progress semaphore PB addr=0x%llx payload=0x%x execute=0x%08x wfi=%d progress_pb=0x%llx",
               (unsigned long long)host_sem_va, host_payload, sem_execute,
               progress_wfi ? 1 : 0, (unsigned long long)progress_va);
    }

    rm_write_gpfifo_entry(gp_words, entry, pb_va, pb_count * sizeof(rm_u32));
    if (progress_words != NULL && progress_va != 0 && progress_count != 0) {
        rm_write_gpfifo_entry(gp_words, progress_entry, progress_va,
                              progress_count * sizeof(rm_u32));
    } else {
        gp_words[progress_entry * 2U] = 0;
        gp_words[progress_entry * 2U + 1U] = 0;
    }
    __sync_synchronize();
    rm_u32 old_put = g.rm_gpfifo_put;
    g.rm_gpfifo_put = (entry + 2U) % 32U;
    if (completion != NULL) {
        completion->gpfifo_put = g.rm_gpfifo_put;
    }
    rm_channel_kickoff_locked(old_put, g.rm_gpfifo_put);
    tracef("RM submitted compute PB channel=0x%x compute=0x%x class=0x%x put=%u entry=%u/%u words=%u/%u pb=0x%llx progress=0x%llx token=0x%08x doorbell=0x%08x qmd_ready=%d qmd_requested=%d qmd_submitted=%d",
           g.rm_channel, g.rm_compute, g.rm_compute_class, g.rm_gpfifo_put,
           entry, progress_entry, pb_count, progress_count,
           (unsigned long long)pb_va, (unsigned long long)progress_va,
           g.rm_work_submit_token, g.rm_doorbell_token,
           qmd_ready ? 1 : 0, qmd_requested ? 1 : 0,
           completion != NULL ? (int)completion->qmd_submitted : 0);
    bool must_wait_completion = env_enabled("LANXIN_NVIDIA_CUDA_WAIT_COMPLETION") ||
                                (qmd_requested && env_enabled("LANXIN_NVIDIA_CUDA_STRICT_LAUNCH"));
    if (qmd_ready && must_wait_completion) {
        return rm_poll_launch_completion_locked(f, qmd_requested);
    }
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
        update_process_memory_accounting_locked();
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
    if (ptr < 4096U || (bytes != 0 && ptr > (CUdeviceptr)(UINTPTR_MAX - bytes))) {
        return NULL;
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
                update_process_memory_accounting_locked();
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
    if (!env_enabled("LANXIN_NVIDIA_CUDA_HOST_ALLOC_ONLY")) {
        struct allocation *rm_alloc = calloc(1, sizeof(*rm_alloc));
        if (rm_alloc != NULL) {
            pthread_mutex_lock(&g_lock);
            if (rm_alloc_mapped_system_locked(bytesize, rm_alloc) == 0) {
                rm_alloc->id = g.next_id++;
                rm_alloc->next = g.allocs;
                g.allocs = rm_alloc;
                g.allocated_bytes += bytesize;
                update_process_memory_accounting_locked();
                *dptr = rm_alloc->dptr;
                pthread_mutex_unlock(&g_lock);
                tracef("cuMemAlloc_v2 RM-backed %zu -> 0x%llx", bytesize,
                       (unsigned long long)*dptr);
                return CUDA_SUCCESS;
            }
            pthread_mutex_unlock(&g_lock);
            free(rm_alloc);
        }
    } else {
        tracef("cuMemAlloc_v2 host-only path %zu", bytesize);
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
    module->image_hash = image != NULL && image_size != 0 ? fnv1a64_bytes(image, image_size) : fnv1a64_str(name);
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
    (void)module_discover_cubin_kernels(module);
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
    const char *image_kind = NULL;
    const void *resolved = resolve_module_image(image, &image_size, &image_kind);
    CUmodule m = make_module("<image>", resolved, image_size, image_size != 0);
    if (m == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    tracef("cuModuleLoadData image=%p resolved=%p kind=%s size=%zu hash=0x%016llx",
           image, resolved, image_kind != NULL ? image_kind : "unknown", image_size,
           (unsigned long long)m->image_hash);
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
    for (struct CUfunc_st *fn = hmod->functions; fn != NULL; fn = fn->next) {
        rm_release_function_launch_locked(fn);
    }
    rm_stage_release_locked(&hmod->code);
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
    return module_get_or_create_function(hfunc, hmod, name);
}

CUresult CUDAAPI cuModuleGetFunctionCount(unsigned int *count, CUmodule mod)
{
    if (count == NULL || !valid_module(mod)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = module_discover_cubin_kernels(mod);
    if (result != CUDA_SUCCESS) {
        return result;
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
        *pi = hfunc->metadata.valid && hfunc->metadata.max_threads_per_block != 0 ?
              (int)hfunc->metadata.max_threads_per_block : 1024;
        return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
        *pi = hfunc->metadata.valid ? (int)hfunc->metadata.static_shared_mem_bytes : 0;
        return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
        *pi = hfunc->metadata.valid ? (int)hfunc->metadata.const_mem_bytes : 0;
        return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
        *pi = hfunc->metadata.valid ? (int)hfunc->metadata.local_mem_bytes : 0;
        return CUDA_SUCCESS;
    case CU_FUNC_ATTRIBUTE_NUM_REGS:
        *pi = hfunc->metadata.valid ? (int)hfunc->metadata.reg_count : 0;
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

static CUkernel library_find_kernel(CUlibrary library, const char *name)
{
    if (!valid_library(library) || name == NULL) {
        return NULL;
    }
    for (CUkernel kernel = library->kernels; kernel != NULL; kernel = kernel->next) {
        if (strcmp(kernel->name, name) == 0) {
            return kernel;
        }
    }
    return NULL;
}

static CUresult library_get_or_create_kernel_for_function(CUkernel *pKernel,
                                                          CUlibrary library,
                                                          CUfunction function)
{
    if (pKernel == NULL || !valid_library(library) || !valid_function(function) ||
        function->name == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUkernel existing = library_find_kernel(library, function->name);
    if (existing != NULL) {
        *pKernel = existing;
        return CUDA_SUCCESS;
    }
    CUkernel kernel = calloc(1, sizeof(*kernel));
    if (kernel == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    kernel->name = strdup(function->name);
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

static CUresult library_populate_kernels(CUlibrary library)
{
    if (!valid_library(library) || !valid_module(library->module)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = module_discover_cubin_kernels(library->module);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    for (struct CUfunc_st *fn = library->module->functions; fn != NULL; fn = fn->next) {
        CUkernel kernel = NULL;
        result = library_get_or_create_kernel_for_function(&kernel, library, fn);
        if (result != CUDA_SUCCESS) {
            return result;
        }
    }
    return CUDA_SUCCESS;
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
    CUresult result = library_populate_kernels(lib);
    if (result != CUDA_SUCCESS) {
        lib->magic = 0;
        free(lib);
        return result;
    }
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
    CUkernel existing = library_find_kernel(library, name);
    if (existing != NULL) {
        *pKernel = existing;
        return CUDA_SUCCESS;
    }
    CUfunction function = NULL;
    CUresult result = cuModuleGetFunction(&function, library->module, name);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    return library_get_or_create_kernel_for_function(pKernel, library, function);
}

CUresult CUDAAPI cuLibraryGetKernelCount(unsigned int *count, CUlibrary lib)
{
    if (count == NULL || !valid_library(lib)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = library_populate_kernels(lib);
    if (result != CUDA_SUCCESS) {
        return result;
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
    CUresult result = library_populate_kernels(lib);
    if (result != CUDA_SUCCESS) {
        return result;
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

static CUresult capture_launch_params(void **kernelParams, void **extra,
                                      void **params_out, size_t *params_size_out, rm_u32 *flags_out)
{
    if (params_out == NULL || params_size_out == NULL || flags_out == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *params_out = NULL;
    *params_size_out = 0;
    *flags_out = 0;
    size_t max_bytes = (size_t)env_ull("LANXIN_NVIDIA_CUDA_PARAM_STAGE_MAX_BYTES", 64ULL * 1024ULL);

    if (extra != NULL) {
        void *buffer = NULL;
        size_t buffer_size = 0;
        for (size_t i = 0; i < 128 && extra[i] != NULL && extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
            void *key = extra[i];
            void *value = extra[i + 1];
            if (key == CU_LAUNCH_PARAM_BUFFER_POINTER) {
                buffer = value;
            } else if (key == CU_LAUNCH_PARAM_BUFFER_SIZE && value != NULL) {
                buffer_size = *(size_t *)value;
            }
        }
        if (buffer != NULL && buffer_size != 0) {
            if (buffer_size > max_bytes) {
                buffer_size = max_bytes;
            }
            void *copy = malloc(buffer_size);
            if (copy == NULL) {
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            memcpy(copy, buffer, buffer_size);
            *params_out = copy;
            *params_size_out = buffer_size;
            *flags_out = LANXIN_LAUNCH_PARAM_EXTRA_BUFFER;
            return CUDA_SUCCESS;
        }
    }

    unsigned long long param_ptr_count = env_ull("LANXIN_NVIDIA_CUDA_KERNEL_PARAM_PTRS", 0);
    if (kernelParams != NULL && param_ptr_count != 0) {
        if (param_ptr_count > 256) {
            param_ptr_count = 256;
        }
        size_t bytes = (size_t)param_ptr_count * sizeof(uintptr_t);
        if (bytes > max_bytes) {
            bytes = max_bytes - (max_bytes % sizeof(uintptr_t));
        }
        if (bytes != 0) {
            void *copy = calloc(1, bytes);
            if (copy == NULL) {
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            uintptr_t *values = (uintptr_t *)copy;
            size_t n = bytes / sizeof(uintptr_t);
            for (size_t i = 0; i < n; i++) {
                values[i] = (uintptr_t)kernelParams[i];
            }
            *params_out = copy;
            *params_size_out = bytes;
            *flags_out = LANXIN_LAUNCH_PARAM_POINTER_ARRAY;
            return CUDA_SUCCESS;
        }
    }
    return CUDA_SUCCESS;
}

struct lanxin_kernel_uint3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

struct lanxin_block_q8_1 {
    uint16_t d;
    uint16_t s;
    int8_t qs[32];
};

struct lanxin_block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};

struct lanxin_rope_corr_dims {
    float v[2];
};

struct lanxin_mrope_sections {
    int v[4];
};

static bool launch_arg_copy(void **kernelParams, size_t index, void *dst, size_t size)
{
    if (kernelParams == NULL || kernelParams[index] == NULL || dst == NULL || size == 0) {
        return false;
    }
    memcpy(dst, kernelParams[index], size);
    return true;
}

static uintptr_t launch_arg_ptr(void **kernelParams, size_t index)
{
    uintptr_t value = 0;
    launch_arg_copy(kernelParams, index, &value, sizeof(value));
    return value;
}

static int launch_arg_i32(void **kernelParams, size_t index)
{
    int value = 0;
    launch_arg_copy(kernelParams, index, &value, sizeof(value));
    return value;
}

static int64_t launch_arg_i64(void **kernelParams, size_t index)
{
    int64_t value = 0;
    launch_arg_copy(kernelParams, index, &value, sizeof(value));
    return value;
}

static uint32_t launch_arg_u32(void **kernelParams, size_t index)
{
    uint32_t value = 0;
    launch_arg_copy(kernelParams, index, &value, sizeof(value));
    return value;
}

static size_t launch_arg_size(void **kernelParams, size_t index)
{
    size_t value = 0;
    launch_arg_copy(kernelParams, index, &value, sizeof(value));
    return value;
}

static float launch_arg_f32(void **kernelParams, size_t index)
{
    float value = 0.0f;
    launch_arg_copy(kernelParams, index, &value, sizeof(value));
    return value;
}

static struct lanxin_kernel_uint3 launch_arg_uint3(void **kernelParams, size_t index)
{
    struct lanxin_kernel_uint3 value = {0, 0, 0};
    launch_arg_copy(kernelParams, index, &value, sizeof(value));
    return value;
}

static uint32_t packed_mod_u32(uint32_t value, struct lanxin_kernel_uint3 packed)
{
    return packed.z == 0 ? 0 : value % packed.z;
}

static uint32_t packed_div_u32(uint32_t value, struct lanxin_kernel_uint3 packed)
{
    return packed.z == 0 ? 0 : value / packed.z;
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

static int8_t lanxin_round_to_i8(float value)
{
    int q = (int)(value >= 0.0f ? value + 0.5f : value - 0.5f);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

static bool resolve_kernel_ptr_locked(uintptr_t ptr, size_t bytes, void **host_out)
{
    if (host_out == NULL) {
        return false;
    }
    *host_out = resolve_uva_ptr_locked((CUdeviceptr)ptr, bytes);
    return *host_out != NULL;
}

static bool i64_mul_overflows_size(int64_t a, int64_t b, size_t elem_size, size_t *out)
{
    if (out == NULL || a < 0 || b < 0 || elem_size == 0) {
        return true;
    }
    uint64_t ua = (uint64_t)a;
    uint64_t ub = (uint64_t)b;
    if (ua != 0 && ub > UINT64_MAX / ua) {
        return true;
    }
    uint64_t prod = ua * ub;
    if (prod > UINT64_MAX / elem_size) {
        return true;
    }
    prod *= elem_size;
    if (prod > (uint64_t)SIZE_MAX) {
        return true;
    }
    *out = (size_t)prod;
    return false;
}

static CUresult host_fallback_compute_batched_ptrs_locked(CUfunction f,
                                                          unsigned int gridDimX, unsigned int gridDimY,
                                                          unsigned int blockDimX, unsigned int blockDimY,
                                                          void **kernelParams)
{
    (void)gridDimX;
    (void)gridDimY;
    (void)blockDimX;
    (void)blockDimY;

    char *src0 = NULL;
    char *src1 = NULL;
    char *dst = NULL;
    const void **ptrs_src = NULL;
    void **ptrs_dst = NULL;

    int64_t ne12 = launch_arg_i64(kernelParams, 5);
    int64_t ne13 = launch_arg_i64(kernelParams, 6);
    int64_t ne23 = launch_arg_i64(kernelParams, 7);
    size_t nb02 = launch_arg_size(kernelParams, 8);
    size_t nb03 = launch_arg_size(kernelParams, 9);
    size_t nb12 = launch_arg_size(kernelParams, 10);
    size_t nb13 = launch_arg_size(kernelParams, 11);
    size_t nbd2 = launch_arg_size(kernelParams, 12);
    size_t nbd3 = launch_arg_size(kernelParams, 13);
    int64_t r2 = launch_arg_i64(kernelParams, 14);
    int64_t r3 = launch_arg_i64(kernelParams, 15);

    size_t ptrs_src_bytes = 0;
    size_t ptrs_dst_bytes = 0;
    if (ne12 < 0 || ne13 < 0 || ne23 < 0 || r2 <= 0 || r3 <= 0 ||
        (ne12 != 0 && ne13 > INT64_MAX / ne12) ||
        ne12 * ne13 != ne23 ||
        i64_mul_overflows_size(ne23, 2, sizeof(void *), &ptrs_src_bytes) ||
        i64_mul_overflows_size(ne23, 1, sizeof(void *), &ptrs_dst_bytes)) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (!resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 0), 0, (void **)&src0) ||
        !resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 1), 0, (void **)&src1) ||
        !resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 2), 0, (void **)&dst) ||
        !resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 3), ptrs_src_bytes, (void **)&ptrs_src) ||
        !resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 4), ptrs_dst_bytes, (void **)&ptrs_dst)) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        int64_t i03 = i13 / r3;
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            int64_t i02 = i12 / r2;
            int64_t idx = i12 + i13 * ne12;
            ptrs_src[(size_t)idx] = src0 + (size_t)i02 * nb02 + (size_t)i03 * nb03;
            ptrs_src[(size_t)ne23 + (size_t)idx] = src1 + (size_t)i12 * nb12 + (size_t)i13 * nb13;
            ptrs_dst[(size_t)idx] = dst + (size_t)i12 * nbd2 + (size_t)i13 * nbd3;
        }
    }

    tracef("host fallback kernel %s ne12=%lld ne13=%lld ne23=%lld",
           f->name != NULL ? f->name : "<unnamed>",
           (long long)ne12, (long long)ne13, (long long)ne23);
    return CUDA_SUCCESS;
}

static CUresult host_fallback_bin_bcast_locked(CUfunction f,
                                               unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                               void **kernelParams)
{
    (void)gridDimX;
    (void)gridDimY;
    (void)gridDimZ;
    const char *name = f->name != NULL ? f->name : "";
    bool is_mul = strstr(name, "op_mulff") != NULL;
    bool is_add = strstr(name, "op_addff") != NULL;
    if (!is_mul && !is_add) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    float *src0 = NULL;
    float *src1 = NULL;
    float *dst = NULL;
    if (!resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 0), 0, (void **)&src0) ||
        !resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 1), 0, (void **)&src1) ||
        !resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 2), 0, (void **)&dst)) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    int ne0 = launch_arg_i32(kernelParams, 3);
    int ne1 = launch_arg_i32(kernelParams, 4);
    int ne2 = launch_arg_i32(kernelParams, 5);
    struct lanxin_kernel_uint3 ne3 = launch_arg_uint3(kernelParams, 6);
    struct lanxin_kernel_uint3 ne10 = launch_arg_uint3(kernelParams, 7);
    struct lanxin_kernel_uint3 ne11 = launch_arg_uint3(kernelParams, 8);
    struct lanxin_kernel_uint3 ne12 = launch_arg_uint3(kernelParams, 9);
    struct lanxin_kernel_uint3 ne13 = launch_arg_uint3(kernelParams, 10);
    int64_t s1 = launch_arg_i32(kernelParams, 11);
    int64_t s2 = launch_arg_i32(kernelParams, 12);
    int64_t s3 = launch_arg_i32(kernelParams, 13);
    int64_t s00 = launch_arg_i32(kernelParams, 14);
    int64_t s01 = launch_arg_i32(kernelParams, 15);
    int64_t s02 = launch_arg_i32(kernelParams, 16);
    int64_t s03 = launch_arg_i32(kernelParams, 17);
    int64_t s10 = launch_arg_i32(kernelParams, 18);
    int64_t s11 = launch_arg_i32(kernelParams, 19);
    int64_t s12 = launch_arg_i32(kernelParams, 20);
    int64_t s13 = launch_arg_i32(kernelParams, 21);
    uint32_t n3 = ne3.z == 0 ? 1 : ne3.z;

    for (uint32_t i3 = 0; i3 < n3; i3++) {
        uint32_t i13 = packed_mod_u32(i3, ne13);
        for (int i2 = 0; i2 < ne2; i2++) {
            uint32_t i12 = packed_mod_u32((uint32_t)i2, ne12);
            for (int i1 = 0; i1 < ne1; i1++) {
                uint32_t i11 = packed_mod_u32((uint32_t)i1, ne11);
                int64_t src0_base = (int64_t)i3 * s03 + (int64_t)i2 * s02 + (int64_t)i1 * s01;
                int64_t src1_base = (int64_t)i13 * s13 + (int64_t)i12 * s12 + (int64_t)i11 * s11;
                int64_t dst_base = (int64_t)i3 * s3 + (int64_t)i2 * s2 + (int64_t)i1 * s1;
                for (int i0 = 0; i0 < ne0; i0++) {
                    uint32_t i10 = packed_mod_u32((uint32_t)i0, ne10);
                    float a = src0[src0_base + (int64_t)i0 * s00];
                    float b = src1[src1_base + (int64_t)i10 * s10];
                    dst[dst_base + i0] = is_mul ? a * b : a + b;
                }
            }
        }
    }
    tracef("host fallback kernel %s", name);
    return CUDA_SUCCESS;
}

static CUresult host_fallback_rms_norm_locked(CUfunction f,
                                              unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                              void **kernelParams)
{
    const char *name = f->name != NULL ? f->name : "";
    float *x = NULL;
    float *dst = NULL;
    if (!resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 0), 0, (void **)&x) ||
        !resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 1), 0, (void **)&dst)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    int ncols = launch_arg_i32(kernelParams, 2);
    int64_t stride_row = launch_arg_i64(kernelParams, 3);
    int64_t stride_channel = launch_arg_i64(kernelParams, 4);
    int64_t stride_sample = launch_arg_i64(kernelParams, 5);
    float eps = launch_arg_f32(kernelParams, 6);
    float *mul = NULL;
    float *add = NULL;
    uintptr_t mul_ptr = launch_arg_ptr(kernelParams, 7);
    uintptr_t add_ptr = launch_arg_ptr(kernelParams, 15);
    if (mul_ptr != 0) {
        resolve_kernel_ptr_locked(mul_ptr, 0, (void **)&mul);
    }
    if (add_ptr != 0) {
        resolve_kernel_ptr_locked(add_ptr, 0, (void **)&add);
    }
    int64_t mul_stride_row = launch_arg_i64(kernelParams, 8);
    int64_t mul_stride_channel = launch_arg_i64(kernelParams, 9);
    int64_t mul_stride_sample = launch_arg_i64(kernelParams, 10);
    struct lanxin_kernel_uint3 mul_ncols = launch_arg_uint3(kernelParams, 11);
    struct lanxin_kernel_uint3 mul_nrows = launch_arg_uint3(kernelParams, 12);
    struct lanxin_kernel_uint3 mul_nchannels = launch_arg_uint3(kernelParams, 13);
    struct lanxin_kernel_uint3 mul_nsamples = launch_arg_uint3(kernelParams, 14);
    int64_t add_stride_row = launch_arg_i64(kernelParams, 16);
    int64_t add_stride_channel = launch_arg_i64(kernelParams, 17);
    int64_t add_stride_sample = launch_arg_i64(kernelParams, 18);
    struct lanxin_kernel_uint3 add_ncols = launch_arg_uint3(kernelParams, 19);
    struct lanxin_kernel_uint3 add_nrows = launch_arg_uint3(kernelParams, 20);
    struct lanxin_kernel_uint3 add_nchannels = launch_arg_uint3(kernelParams, 21);
    struct lanxin_kernel_uint3 add_nsamples = launch_arg_uint3(kernelParams, 22);

    for (unsigned int sample = 0; sample < gridDimZ; sample++) {
        for (unsigned int channel = 0; channel < gridDimY; channel++) {
            for (unsigned int row = 0; row < gridDimX; row++) {
                float *xb = x + (int64_t)sample * stride_sample + (int64_t)channel * stride_channel + (int64_t)row * stride_row;
                float *db = dst + (((int64_t)sample * gridDimY + channel) * gridDimX + row) * ncols;
                double sum = 0.0;
                for (int col = 0; col < ncols; col++) {
                    sum += (double)xb[col] * (double)xb[col];
                }
                float scale = 1.0f / sqrtf((float)(sum / (double)ncols) + eps);
                float *mb = NULL;
                float *ab = NULL;
                if (mul != NULL) {
                    mb = mul + (int64_t)packed_mod_u32(sample, mul_nsamples) * mul_stride_sample +
                         (int64_t)packed_mod_u32(channel, mul_nchannels) * mul_stride_channel +
                         (int64_t)packed_mod_u32(row, mul_nrows) * mul_stride_row;
                }
                if (add != NULL) {
                    ab = add + (int64_t)packed_mod_u32(sample, add_nsamples) * add_stride_sample +
                         (int64_t)packed_mod_u32(channel, add_nchannels) * add_stride_channel +
                         (int64_t)packed_mod_u32(row, add_nrows) * add_stride_row;
                }
                for (int col = 0; col < ncols; col++) {
                    float value = xb[col] * scale;
                    if (mb != NULL) {
                        value *= mb[packed_mod_u32((uint32_t)col, mul_ncols)];
                    }
                    if (ab != NULL) {
                        value += ab[packed_mod_u32((uint32_t)col, add_ncols)];
                    }
                    db[col] = value;
                }
            }
        }
    }
    tracef("host fallback kernel %s", name);
    return CUDA_SUCCESS;
}

static CUresult host_fallback_get_rows_float_locked(CUfunction f,
                                                    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                                    unsigned int blockDimX, void **kernelParams)
{
    (void)gridDimY;
    (void)gridDimZ;
    (void)blockDimX;
    const char *name = f->name != NULL ? f->name : "";
    float *src0 = NULL;
    int32_t *src1 = NULL;
    float *dst = NULL;
    if (!resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 0), 0, (void **)&src0) ||
        !resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 1), 0, (void **)&src1) ||
        !resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 2), 0, (void **)&dst)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    int64_t ne00 = launch_arg_i64(kernelParams, 3);
    int64_t ne11 = launch_arg_i64(kernelParams, 4);
    struct lanxin_kernel_uint3 ne12 = launch_arg_uint3(kernelParams, 5);
    size_t s1 = (size_t)launch_arg_i64(kernelParams, 6);
    size_t s2 = (size_t)launch_arg_i64(kernelParams, 7);
    size_t s3 = (size_t)launch_arg_i64(kernelParams, 8);
    size_t nb01 = (size_t)launch_arg_i64(kernelParams, 9);
    size_t nb02 = (size_t)launch_arg_i64(kernelParams, 10);
    size_t nb03 = (size_t)launch_arg_i64(kernelParams, 11);
    size_t s10 = (size_t)launch_arg_i64(kernelParams, 12);
    size_t s11 = (size_t)launch_arg_i64(kernelParams, 13);
    size_t s12 = (size_t)launch_arg_i64(kernelParams, 14);
    uint32_t n12 = ne12.z == 0 ? 1 : ne12.z;

    for (unsigned int i10 = 0; i10 < gridDimX; i10++) {
        for (int64_t z = 0; z < ne11 * (int64_t)n12; z++) {
            int64_t i11 = z / (int64_t)n12;
            int64_t i12 = z - i11 * (int64_t)n12;
            int i01 = src1[(size_t)i10 * s10 + (size_t)i11 * s11 + (size_t)i12 * s12];
            float *dst_row = dst + (size_t)i10 * s1 + (size_t)i11 * s2 + (size_t)i12 * s3;
            float *src_row = (float *)((char *)src0 + (size_t)i01 * nb01 + (size_t)i11 * nb02 + (size_t)i12 * nb03);
            for (int64_t i00 = 0; i00 < ne00; i00++) {
                dst_row[i00] = src_row[i00];
            }
        }
    }
    tracef("host fallback kernel %s", name);
    return CUDA_SUCCESS;
}

static CUresult host_fallback_quantize_q8_1_locked(CUfunction f,
                                                   unsigned int gridDimY, unsigned int gridDimZ,
                                                   void **kernelParams)
{
    const char *name = f->name != NULL ? f->name : "";
    float *x = NULL;
    struct lanxin_block_q8_1 *y = NULL;
    if (!resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 0), 0, (void **)&x) ||
        !resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 1), 0, (void **)&y)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    int64_t ne00 = launch_arg_i64(kernelParams, 2);
    int64_t s01 = launch_arg_i64(kernelParams, 3);
    int64_t s02 = launch_arg_i64(kernelParams, 4);
    int64_t s03 = launch_arg_i64(kernelParams, 5);
    int64_t ne0 = launch_arg_i64(kernelParams, 6);
    uint32_t ne1 = launch_arg_u32(kernelParams, 7);
    struct lanxin_kernel_uint3 ne2 = launch_arg_uint3(kernelParams, 8);
    uint32_t n2 = ne2.z == 0 ? 1 : ne2.z;

    for (uint32_t bz = 0; bz < gridDimZ; bz++) {
        int64_t i3 = (int64_t)(bz / n2);
        int64_t i2 = (int64_t)(bz - (uint32_t)i3 * n2);
        for (uint32_t i1 = 0; i1 < gridDimY && i1 < ne1; i1++) {
            for (int64_t block0 = 0; block0 < ne0; block0 += 32) {
                float amax = 0.0f;
                float sum = 0.0f;
                float vals[32];
                for (int j = 0; j < 32; j++) {
                    int64_t i0 = block0 + j;
                    float v = i0 < ne00 ? x[i3 * s03 + i2 * s02 + (int64_t)i1 * s01 + i0] : 0.0f;
                    vals[j] = v;
                    float av = v < 0.0f ? -v : v;
                    if (av > amax) {
                        amax = av;
                    }
                    sum += v;
                }
                float d = amax / 127.0f;
                float id = d != 0.0f ? 1.0f / d : 0.0f;
                int64_t i_cont = ((i3 * (int64_t)n2 + i2) * (int64_t)ne1 + (int64_t)i1) * ne0 + block0;
                struct lanxin_block_q8_1 *yb = &y[i_cont / 32];
                yb->d = lanxin_float_to_half(d);
                yb->s = lanxin_float_to_half(sum * d);
                for (int j = 0; j < 32; j++) {
                    yb->qs[j] = lanxin_round_to_i8(vals[j] * id);
                }
            }
        }
    }
    tracef("host fallback kernel %s", name);
    return CUDA_SUCCESS;
}

static CUresult host_fallback_torch_fill_locked(CUfunction f, void **kernelParams)
{
    const char *name = f->name != NULL ? f->name : "";
    if (strstr(name, "FillFunctor") == NULL ||
        strstr(name, "St5arrayIPcLm1") == NULL) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    int n = launch_arg_i32(kernelParams, 0);
    if (n < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    uintptr_t dst_ptr = launch_arg_ptr(kernelParams, 2);
    if (dst_ptr == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (strstr(name, "FillFunctorIi") != NULL) {
        int32_t *dst = NULL;
        int32_t value = 0;
        launch_arg_copy(kernelParams, 1, &value, sizeof(value));
        if (!resolve_kernel_ptr_locked(dst_ptr, (size_t)n * sizeof(*dst), (void **)&dst)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        for (int i = 0; i < n; i++) {
            dst[i] = value;
        }
        tracef("host fallback kernel %s n=%d value=%d", name, n, (int)value);
        return CUDA_SUCCESS;
    }
    if (strstr(name, "FillFunctorIl") != NULL) {
        int64_t *dst = NULL;
        int64_t value = 0;
        launch_arg_copy(kernelParams, 1, &value, sizeof(value));
        if (!resolve_kernel_ptr_locked(dst_ptr, (size_t)n * sizeof(*dst), (void **)&dst)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        for (int i = 0; i < n; i++) {
            dst[i] = value;
        }
        tracef("host fallback kernel %s n=%d", name, n);
        return CUDA_SUCCESS;
    }
    if (strstr(name, "FillFunctorIN3c104HalfE") != NULL ||
        strstr(name, "FillFunctorIN3c108BFloat16E") != NULL ||
        strstr(name, "FillFunctorIs") != NULL ||
        strstr(name, "FillFunctorIt") != NULL) {
        uint16_t *dst = NULL;
        uint16_t value = 0;
        launch_arg_copy(kernelParams, 1, &value, sizeof(value));
        if (!resolve_kernel_ptr_locked(dst_ptr, (size_t)n * sizeof(*dst), (void **)&dst)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        for (int i = 0; i < n; i++) {
            dst[i] = value;
        }
        tracef("host fallback kernel %s n=%d value_bits=0x%04x", name, n, (unsigned)value);
        return CUDA_SUCCESS;
    }
    if (strstr(name, "FillFunctorIa") != NULL ||
        strstr(name, "FillFunctorIh") != NULL) {
        uint8_t *dst = NULL;
        uint8_t value = 0;
        launch_arg_copy(kernelParams, 1, &value, sizeof(value));
        if (!resolve_kernel_ptr_locked(dst_ptr, (size_t)n * sizeof(*dst), (void **)&dst)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        memset(dst, value, (size_t)n);
        tracef("host fallback kernel %s n=%d value=0x%02x", name, n, (unsigned)value);
        return CUDA_SUCCESS;
    }
    if (strstr(name, "FillFunctorIj") != NULL) {
        uint32_t *dst = NULL;
        uint32_t value = 0;
        launch_arg_copy(kernelParams, 1, &value, sizeof(value));
        if (!resolve_kernel_ptr_locked(dst_ptr, (size_t)n * sizeof(*dst), (void **)&dst)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        for (int i = 0; i < n; i++) {
            dst[i] = value;
        }
        tracef("host fallback kernel %s n=%d value=%u", name, n, (unsigned)value);
        return CUDA_SUCCESS;
    }
    if (strstr(name, "FillFunctorIm") != NULL) {
        uint64_t *dst = NULL;
        uint64_t value = 0;
        launch_arg_copy(kernelParams, 1, &value, sizeof(value));
        if (!resolve_kernel_ptr_locked(dst_ptr, (size_t)n * sizeof(*dst), (void **)&dst)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        for (int i = 0; i < n; i++) {
            dst[i] = value;
        }
        tracef("host fallback kernel %s n=%d", name, n);
        return CUDA_SUCCESS;
    }
    if (strstr(name, "FillFunctorIf") != NULL) {
        float *dst = NULL;
        float value = 0.0f;
        launch_arg_copy(kernelParams, 1, &value, sizeof(value));
        if (!resolve_kernel_ptr_locked(dst_ptr, (size_t)n * sizeof(*dst), (void **)&dst)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        for (int i = 0; i < n; i++) {
            dst[i] = value;
        }
        tracef("host fallback kernel %s n=%d value=%f", name, n, value);
        return CUDA_SUCCESS;
    }
    if (strstr(name, "FillFunctorId") != NULL) {
        double *dst = NULL;
        double value = 0.0;
        launch_arg_copy(kernelParams, 1, &value, sizeof(value));
        if (!resolve_kernel_ptr_locked(dst_ptr, (size_t)n * sizeof(*dst), (void **)&dst)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        for (int i = 0; i < n; i++) {
            dst[i] = value;
        }
        tracef("host fallback kernel %s n=%d", name, n);
        return CUDA_SUCCESS;
    }
    if (strstr(name, "FillFunctorIb") != NULL) {
        uint8_t *dst = NULL;
        uint8_t value = 0;
        launch_arg_copy(kernelParams, 1, &value, sizeof(value));
        if (!resolve_kernel_ptr_locked(dst_ptr, (size_t)n * sizeof(*dst), (void **)&dst)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        memset(dst, value ? 1 : 0, (size_t)n);
        tracef("host fallback kernel %s n=%d value=%u", name, n, (unsigned)value);
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

struct lanxin_torch_cast_policy {
    int8_t dtype;
    uint8_t reserved[3];
    uint32_t element_size;
};

static float lanxin_bfloat16_to_float(uint16_t value)
{
    uint32_t bits = (uint32_t)value << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t lanxin_float_to_bfloat16(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1U;
    uint32_t rounding_bias = 0x7fffU + lsb;
    return (uint16_t)((bits + rounding_bias) >> 16);
}

static bool torch_dtype_supported_for_direct_copy(int dtype, uint32_t element_size)
{
    switch (dtype) {
    case LANXIN_TORCH_DTYPE_BYTE:
    case LANXIN_TORCH_DTYPE_CHAR:
    case LANXIN_TORCH_DTYPE_BOOL:
        return element_size == 1;
    case LANXIN_TORCH_DTYPE_SHORT:
    case LANXIN_TORCH_DTYPE_HALF:
    case LANXIN_TORCH_DTYPE_BFLOAT16:
    case LANXIN_TORCH_DTYPE_UINT16:
        return element_size == 2;
    case LANXIN_TORCH_DTYPE_INT:
    case LANXIN_TORCH_DTYPE_FLOAT:
    case LANXIN_TORCH_DTYPE_UINT32:
        return element_size == 4;
    case LANXIN_TORCH_DTYPE_LONG:
    case LANXIN_TORCH_DTYPE_DOUBLE:
    case LANXIN_TORCH_DTYPE_UINT64:
        return element_size == 8;
    default:
        return false;
    }
}

static double torch_direct_read_scalar_as_double(const void *base, size_t index,
                                                 int dtype, uint32_t element_size)
{
    const char *ptr = (const char *)base + index * (size_t)element_size;
    switch (dtype) {
    case LANXIN_TORCH_DTYPE_BYTE:
        return (double)*(const uint8_t *)ptr;
    case LANXIN_TORCH_DTYPE_CHAR:
        return (double)*(const int8_t *)ptr;
    case LANXIN_TORCH_DTYPE_SHORT:
        return (double)*(const int16_t *)ptr;
    case LANXIN_TORCH_DTYPE_INT:
        return (double)*(const int32_t *)ptr;
    case LANXIN_TORCH_DTYPE_LONG:
        return (double)*(const int64_t *)ptr;
    case LANXIN_TORCH_DTYPE_HALF:
        return (double)lanxin_half_to_float(*(const uint16_t *)ptr);
    case LANXIN_TORCH_DTYPE_FLOAT:
        return (double)*(const float *)ptr;
    case LANXIN_TORCH_DTYPE_DOUBLE:
        return *(const double *)ptr;
    case LANXIN_TORCH_DTYPE_BOOL:
        return *(const uint8_t *)ptr != 0 ? 1.0 : 0.0;
    case LANXIN_TORCH_DTYPE_BFLOAT16:
        return (double)lanxin_bfloat16_to_float(*(const uint16_t *)ptr);
    case LANXIN_TORCH_DTYPE_UINT16:
        return (double)*(const uint16_t *)ptr;
    case LANXIN_TORCH_DTYPE_UINT32:
        return (double)*(const uint32_t *)ptr;
    case LANXIN_TORCH_DTYPE_UINT64:
        return (double)*(const uint64_t *)ptr;
    default:
        return 0.0;
    }
}

static int64_t torch_direct_read_scalar_as_i64(const void *base, size_t index,
                                               int dtype, uint32_t element_size)
{
    const char *ptr = (const char *)base + index * (size_t)element_size;
    switch (dtype) {
    case LANXIN_TORCH_DTYPE_BYTE:
        return (int64_t)*(const uint8_t *)ptr;
    case LANXIN_TORCH_DTYPE_CHAR:
        return (int64_t)*(const int8_t *)ptr;
    case LANXIN_TORCH_DTYPE_SHORT:
        return (int64_t)*(const int16_t *)ptr;
    case LANXIN_TORCH_DTYPE_INT:
        return (int64_t)*(const int32_t *)ptr;
    case LANXIN_TORCH_DTYPE_LONG:
        return *(const int64_t *)ptr;
    case LANXIN_TORCH_DTYPE_BOOL:
        return *(const uint8_t *)ptr != 0 ? 1 : 0;
    case LANXIN_TORCH_DTYPE_UINT16:
        return (int64_t)*(const uint16_t *)ptr;
    case LANXIN_TORCH_DTYPE_UINT32:
        return (int64_t)*(const uint32_t *)ptr;
    default:
        return (int64_t)torch_direct_read_scalar_as_double(base, index, dtype,
                                                           element_size);
    }
}

static void torch_direct_write_scalar_from_double(void *base, size_t index,
                                                  int dtype, uint32_t element_size,
                                                  double value)
{
    char *ptr = (char *)base + index * (size_t)element_size;
    switch (dtype) {
    case LANXIN_TORCH_DTYPE_BYTE:
        *(uint8_t *)ptr = (uint8_t)value;
        break;
    case LANXIN_TORCH_DTYPE_CHAR:
        *(int8_t *)ptr = (int8_t)value;
        break;
    case LANXIN_TORCH_DTYPE_SHORT:
        *(int16_t *)ptr = (int16_t)value;
        break;
    case LANXIN_TORCH_DTYPE_INT:
        *(int32_t *)ptr = (int32_t)value;
        break;
    case LANXIN_TORCH_DTYPE_LONG:
        *(int64_t *)ptr = (int64_t)value;
        break;
    case LANXIN_TORCH_DTYPE_HALF:
        *(uint16_t *)ptr = lanxin_float_to_half((float)value);
        break;
    case LANXIN_TORCH_DTYPE_FLOAT:
        *(float *)ptr = (float)value;
        break;
    case LANXIN_TORCH_DTYPE_DOUBLE:
        *(double *)ptr = value;
        break;
    case LANXIN_TORCH_DTYPE_BOOL:
        *(uint8_t *)ptr = value != 0.0 ? 1 : 0;
        break;
    case LANXIN_TORCH_DTYPE_BFLOAT16:
        *(uint16_t *)ptr = lanxin_float_to_bfloat16((float)value);
        break;
    case LANXIN_TORCH_DTYPE_UINT16:
        *(uint16_t *)ptr = (uint16_t)value;
        break;
    case LANXIN_TORCH_DTYPE_UINT32:
        *(uint32_t *)ptr = (uint32_t)value;
        break;
    case LANXIN_TORCH_DTYPE_UINT64:
        *(uint64_t *)ptr = (uint64_t)value;
        break;
    default:
        break;
    }
}

static void torch_direct_write_scalar_from_i64(void *base, size_t index,
                                               int dtype, uint32_t element_size,
                                               int64_t value)
{
    char *ptr = (char *)base + index * (size_t)element_size;
    switch (dtype) {
    case LANXIN_TORCH_DTYPE_BYTE:
        *(uint8_t *)ptr = (uint8_t)value;
        break;
    case LANXIN_TORCH_DTYPE_CHAR:
        *(int8_t *)ptr = (int8_t)value;
        break;
    case LANXIN_TORCH_DTYPE_SHORT:
        *(int16_t *)ptr = (int16_t)value;
        break;
    case LANXIN_TORCH_DTYPE_INT:
        *(int32_t *)ptr = (int32_t)value;
        break;
    case LANXIN_TORCH_DTYPE_LONG:
        *(int64_t *)ptr = value;
        break;
    case LANXIN_TORCH_DTYPE_BOOL:
        *(uint8_t *)ptr = value != 0 ? 1 : 0;
        break;
    case LANXIN_TORCH_DTYPE_UINT16:
        *(uint16_t *)ptr = (uint16_t)value;
        break;
    case LANXIN_TORCH_DTYPE_UINT32:
        *(uint32_t *)ptr = (uint32_t)value;
        break;
    case LANXIN_TORCH_DTYPE_UINT64:
        *(uint64_t *)ptr = (uint64_t)value;
        break;
    default:
        torch_direct_write_scalar_from_double(base, index, dtype, element_size,
                                              (double)value);
        break;
    }
}

static bool torch_dtype_is_integral_like(int dtype)
{
    return dtype == LANXIN_TORCH_DTYPE_BYTE ||
           dtype == LANXIN_TORCH_DTYPE_CHAR ||
           dtype == LANXIN_TORCH_DTYPE_SHORT ||
           dtype == LANXIN_TORCH_DTYPE_INT ||
           dtype == LANXIN_TORCH_DTYPE_LONG ||
           dtype == LANXIN_TORCH_DTYPE_BOOL ||
           dtype == LANXIN_TORCH_DTYPE_UINT16 ||
           dtype == LANXIN_TORCH_DTYPE_UINT32 ||
           dtype == LANXIN_TORCH_DTYPE_UINT64;
}

static CUresult host_fallback_torch_direct_copy_locked(CUfunction f,
                                                       void **kernelParams)
{
    const char *name = f->name != NULL ? f->name : "";
    if (strstr(name, "direct_copy_kernel_cuda") == NULL ||
        strstr(name, "St5arrayIPcLm2E") == NULL ||
        strstr(name, "LoadWithCastILi1EE") == NULL ||
        strstr(name, "StoreWithCastILi1EE") == NULL) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    int n = launch_arg_i32(kernelParams, 0);
    if (n < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    uintptr_t data_ptrs[2] = {0, 0};
    launch_arg_copy(kernelParams, 2, data_ptrs, sizeof(data_ptrs));
    if (data_ptrs[0] == 0 || data_ptrs[1] == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    struct lanxin_torch_cast_policy loader = {0, {0, 0, 0}, 0};
    struct lanxin_torch_cast_policy storer = {0, {0, 0, 0}, 0};
    launch_arg_copy(kernelParams, 5, &loader, sizeof(loader));
    launch_arg_copy(kernelParams, 6, &storer, sizeof(storer));
    if (!torch_dtype_supported_for_direct_copy((int)loader.dtype,
                                               loader.element_size) ||
        !torch_dtype_supported_for_direct_copy((int)storer.dtype,
                                               storer.element_size)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    void *dst = NULL;
    void *src = NULL;
    size_t dst_bytes = (size_t)n * (size_t)storer.element_size;
    size_t src_bytes = (size_t)n * (size_t)loader.element_size;
    if (!resolve_kernel_ptr_locked(data_ptrs[0], dst_bytes, &dst) ||
        !resolve_kernel_ptr_locked(data_ptrs[1], src_bytes, &src)) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (loader.dtype == storer.dtype &&
        loader.element_size == storer.element_size) {
        memmove(dst, src, dst_bytes);
        tracef("host fallback kernel %s n=%d dtype=%d bytes=%zu",
               name, n, (int)storer.dtype, dst_bytes);
        return CUDA_SUCCESS;
    }

    bool integral_path = torch_dtype_is_integral_like((int)loader.dtype) &&
                         torch_dtype_is_integral_like((int)storer.dtype);
    for (int i = 0; i < n; i++) {
        if (integral_path) {
            int64_t value = torch_direct_read_scalar_as_i64(src, (size_t)i,
                                                            (int)loader.dtype,
                                                            loader.element_size);
            torch_direct_write_scalar_from_i64(dst, (size_t)i,
                                               (int)storer.dtype,
                                               storer.element_size, value);
        } else {
            double value = torch_direct_read_scalar_as_double(src, (size_t)i,
                                                              (int)loader.dtype,
                                                              loader.element_size);
            torch_direct_write_scalar_from_double(dst, (size_t)i,
                                                  (int)storer.dtype,
                                                  storer.element_size, value);
        }
    }
    tracef("host fallback kernel %s n=%d src_dtype=%d dst_dtype=%d",
           name, n, (int)loader.dtype, (int)storer.dtype);
    return CUDA_SUCCESS;
}

static CUresult host_fallback_torch_nocast_direct_copy_locked(CUfunction f,
                                                              void **kernelParams)
{
    const char *name = f->name != NULL ? f->name : "";
    if (!env_enabled("LANXIN_NVIDIA_CUDA_GUESS_NOCAST_DIRECT_COPY")) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (strstr(name, "direct_copy_kernel_cuda") == NULL ||
        strstr(name, "gpu_kernel_impl_nocast") == NULL ||
        strstr(name, "N3c104HalfE") == NULL) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    int n = launch_arg_i32(kernelParams, 0);
    if (n < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const int candidate_arg_indices[] = {2, 1, 3};
    for (size_t i = 0; i < sizeof(candidate_arg_indices) / sizeof(candidate_arg_indices[0]); i++) {
        uintptr_t data_ptrs[2] = {0, 0};
        launch_arg_copy(kernelParams, candidate_arg_indices[i], data_ptrs, sizeof(data_ptrs));
        if (data_ptrs[0] == 0 || data_ptrs[1] == 0) {
            continue;
        }

        void *dst = NULL;
        void *src = NULL;
        size_t bytes = (size_t)n * sizeof(uint16_t);
        if (!resolve_kernel_ptr_locked(data_ptrs[0], bytes, &dst) ||
            !resolve_kernel_ptr_locked(data_ptrs[1], bytes, &src)) {
            continue;
        }

        memmove(dst, src, bytes);
        tracef("host fallback kernel %s n=%d dtype=Half bytes=%zu ptr_arg=%d",
               name, n, bytes, candidate_arg_indices[i]);
        return CUDA_SUCCESS;
    }

    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult host_fallback_torch_float_to_lowp_copy_locked(CUfunction f,
                                                              void **kernelParams)
{
    const char *name = f->name != NULL ? f->name : "";
    bool to_bfloat16 = strstr(name, "bfloat16_copy_kernel_cuda") != NULL;
    bool to_half = !to_bfloat16 &&
                   strstr(name, "float16_copy_kernel_cuda") != NULL;
    if ((!to_half && !to_bfloat16) ||
        strstr(name, "vectorized_elementwise_kernel") == NULL ||
        strstr(name, "St5arrayIPcLm2E") == NULL) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    int n = launch_arg_i32(kernelParams, 0);
    if (n < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    uintptr_t data_ptrs[2] = {0, 0};
    launch_arg_copy(kernelParams, 2, data_ptrs, sizeof(data_ptrs));
    if (data_ptrs[0] == 0 || data_ptrs[1] == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    uint16_t *dst = NULL;
    float *src = NULL;
    if (!resolve_kernel_ptr_locked(data_ptrs[0], (size_t)n * sizeof(*dst),
                                   (void **)&dst) ||
        !resolve_kernel_ptr_locked(data_ptrs[1], (size_t)n * sizeof(*src),
                                   (void **)&src)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (int i = 0; i < n; i++) {
        dst[i] = to_half ? lanxin_float_to_half(src[i])
                         : lanxin_float_to_bfloat16(src[i]);
    }
    tracef("host fallback kernel %s n=%d src_dtype=6 dst_dtype=%d",
           name, n, to_half ? LANXIN_TORCH_DTYPE_HALF
                            : LANXIN_TORCH_DTYPE_BFLOAT16);
    return CUDA_SUCCESS;
}

static float rope_yarn_ramp_host(float low, float high, int i0)
{
    float denom = high - low;
    if (denom < 0.001f) {
        denom = 0.001f;
    }
    float y = ((float)i0 / 2.0f - low) / denom;
    if (y < 0.0f) y = 0.0f;
    if (y > 1.0f) y = 1.0f;
    return 1.0f - y;
}

static void rope_yarn_host(bool forward, float theta_extrap, float freq_scale,
                           struct lanxin_rope_corr_dims corr_dims, int i0,
                           float ext_factor, float mscale,
                           float *cos_theta, float *sin_theta)
{
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp_host(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
    if (!forward) {
        *sin_theta *= -1.0f;
    }
}

static CUresult host_fallback_rope_multi_float_locked(CUfunction f,
                                                      unsigned int gridDimX, unsigned int blockDimX,
                                                      void **kernelParams)
{
    const char *name = f->name != NULL ? f->name : "";
    bool forward = strstr(name, "_ZL10rope_multiILb1E") != NULL;
    bool has_ff = strstr(name, "_ZL10rope_multiILb1ELb1E") != NULL ||
                  strstr(name, "_ZL10rope_multiILb0ELb1E") != NULL;
    if (strstr(name, "EfEvPKT1_PS0_") == NULL) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    float *x = NULL;
    float *dst = NULL;
    int32_t *pos = NULL;
    float *freq_factors = NULL;
    if (!resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 0), 0, (void **)&x) ||
        !resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 1), 0, (void **)&dst) ||
        !resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 12), 0, (void **)&pos)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    uintptr_t ff_ptr = launch_arg_ptr(kernelParams, 18);
    if (ff_ptr != 0) {
        resolve_kernel_ptr_locked(ff_ptr, 0, (void **)&freq_factors);
    }
    if (has_ff && freq_factors == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    int ne00 = launch_arg_i32(kernelParams, 2);
    int ne01 = launch_arg_i32(kernelParams, 3);
    int ne02 = launch_arg_i32(kernelParams, 4);
    int s01 = launch_arg_i32(kernelParams, 5);
    int s02 = launch_arg_i32(kernelParams, 6);
    int s03 = launch_arg_i32(kernelParams, 7);
    int s1 = launch_arg_i32(kernelParams, 8);
    int s2 = launch_arg_i32(kernelParams, 9);
    int s3 = launch_arg_i32(kernelParams, 10);
    int n_dims = launch_arg_i32(kernelParams, 11);
    float freq_scale = launch_arg_f32(kernelParams, 13);
    float ext_factor = launch_arg_f32(kernelParams, 14);
    float attn_factor = launch_arg_f32(kernelParams, 15);
    struct lanxin_rope_corr_dims corr_dims = {{0.0f, 0.0f}};
    launch_arg_copy(kernelParams, 16, &corr_dims, sizeof(corr_dims));
    float theta_scale = launch_arg_f32(kernelParams, 17);
    struct lanxin_mrope_sections sections = {{0, 0, 0, 0}};
    launch_arg_copy(kernelParams, 19, &sections, sizeof(sections));
    bool is_imrope = false;
    launch_arg_copy(kernelParams, 20, &is_imrope, sizeof(is_imrope));

    if (ne00 <= 0 || ne01 <= 0 || ne02 <= 0 || n_dims <= 0 || n_dims > ne00) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    int sec_w = sections.v[1] + sections.v[0];
    if (sect_dims <= 0) {
        sections.v[0] = n_dims / 2;
        sections.v[1] = 0;
        sections.v[2] = 0;
        sections.v[3] = 0;
        sect_dims = sections.v[0];
        sec_w = sections.v[0];
    }

    uint32_t rows = gridDimX * (blockDimX == 0 ? 1U : blockDimX);
    for (uint32_t row_dst = 0; row_dst < rows; row_dst++) {
        uint32_t i3 = row_dst / (uint32_t)(ne01 * ne02);
        uint32_t i2 = (row_dst - i3 * (uint32_t)(ne01 * ne02)) / (uint32_t)ne01;
        uint32_t i1 = row_dst - i3 * (uint32_t)(ne01 * ne02) - i2 * (uint32_t)ne01;
        if (i1 >= (uint32_t)ne01 || i2 >= (uint32_t)ne02) {
            continue;
        }
        for (int i0 = 0; i0 < ne00; i0 += 2) {
            int idst = i0 / 2 + (int)i1 * s1 + (int)i2 * s2 + (int)i3 * s3;
            int ix = i0 / 2 + (int)i1 * s01 + (int)i2 * s02 + (int)i3 * s03;
            if (i0 >= n_dims) {
                dst[idst + i0 / 2 + 0] = x[ix + i0 / 2 + 0];
                dst[idst + i0 / 2 + 1] = x[ix + i0 / 2 + 1];
                continue;
            }

            int sector = (i0 / 2) % sect_dims;
            float theta_base = 0.0f;
            float theta_pow = powf(theta_scale, (float)i0 / 2.0f);
            if (is_imrope) {
                if (sector % 3 == 1 && sector < 3 * sections.v[1]) {
                    theta_base = (float)pos[i2 + (uint32_t)ne02 * 1U] * theta_pow;
                } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {
                    theta_base = (float)pos[i2 + (uint32_t)ne02 * 2U] * theta_pow;
                } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {
                    theta_base = (float)pos[i2] * theta_pow;
                } else {
                    theta_base = (float)pos[i2 + (uint32_t)ne02 * 3U] * theta_pow;
                }
            } else {
                if (sector < sections.v[0]) {
                    theta_base = (float)pos[i2] * theta_pow;
                } else if (sector >= sections.v[0] && sector < sec_w) {
                    theta_base = (float)pos[i2 + (uint32_t)ne02 * 1U] * theta_pow;
                } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
                    theta_base = (float)pos[i2 + (uint32_t)ne02 * 2U] * theta_pow;
                } else {
                    theta_base = (float)pos[i2 + (uint32_t)ne02 * 3U] * theta_pow;
                }
            }

            float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;
            float cos_theta = 1.0f;
            float sin_theta = 0.0f;
            rope_yarn_host(forward, theta_base / freq_factor, freq_scale, corr_dims, i0,
                           ext_factor, attn_factor, &cos_theta, &sin_theta);
            float x0 = x[ix + 0];
            float x1 = x[ix + n_dims / 2];
            dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
            dst[idst + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
        }
    }
    tracef("host fallback kernel %s rows=%u ne00=%d n_dims=%d", name, rows, ne00, n_dims);
    return CUDA_SUCCESS;
}

static float q6_k_value(const struct lanxin_block_q6_K *block, int idx)
{
    float d = lanxin_half_to_float(block->d);
    int n = idx < 128 ? 0 : 128;
    int l = idx - n;
    int ql_index;
    int qh_index;
    int shift;
    int scale_index;
    int q;
    if (l < 32) {
        ql_index = n / 2 + l;
        qh_index = n / 4 + l;
        shift = 0;
        scale_index = l / 16 + 0;
        q = (int)((block->ql[ql_index] & 0x0fU) | (((block->qh[qh_index] >> shift) & 3U) << 4)) - 32;
    } else if (l < 64) {
        ql_index = n / 2 + l;
        qh_index = n / 4 + (l - 32);
        shift = 2;
        scale_index = (l - 32) / 16 + 2;
        q = (int)((block->ql[ql_index] & 0x0fU) | (((block->qh[qh_index] >> shift) & 3U) << 4)) - 32;
    } else if (l < 96) {
        ql_index = n / 2 + (l - 64);
        qh_index = n / 4 + (l - 64);
        shift = 4;
        scale_index = (l - 64) / 16 + 4;
        q = (int)(((block->ql[ql_index] >> 4) & 0x0fU) | (((block->qh[qh_index] >> shift) & 3U) << 4)) - 32;
    } else {
        ql_index = n / 2 + (l - 64);
        qh_index = n / 4 + (l - 96);
        shift = 6;
        scale_index = (l - 96) / 16 + 6;
        q = (int)(((block->ql[ql_index] >> 4) & 0x0fU) | (((block->qh[qh_index] >> shift) & 3U) << 4)) - 32;
    }
    return d * (float)block->scales[n == 0 ? scale_index : scale_index + 8] * (float)q;
}

static CUresult host_fallback_mmvq_q6k_ncols1_locked(CUfunction f,
                                                     unsigned int gridDimY, unsigned int gridDimZ,
                                                     void **kernelParams)
{
    const char *name = f->name != NULL ? f->name : "";
    struct lanxin_block_q6_K *vx = NULL;
    struct lanxin_block_q8_1 *vy = NULL;
    int32_t *ids = NULL;
    float *dst = NULL;
    if (!resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 0), 0, (void **)&vx) ||
        !resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 1), 0, (void **)&vy) ||
        !resolve_kernel_ptr_locked(launch_arg_ptr(kernelParams, 4), 0, (void **)&dst)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    uintptr_t ids_ptr = launch_arg_ptr(kernelParams, 2);
    if (ids_ptr != 0) {
        resolve_kernel_ptr_locked(ids_ptr, 0, (void **)&ids);
    }
    uint32_t ncols_x = launch_arg_u32(kernelParams, 5);
    struct lanxin_kernel_uint3 nchannels_y = launch_arg_uint3(kernelParams, 6);
    uint32_t stride_row_x = launch_arg_u32(kernelParams, 7);
    uint32_t stride_col_y = launch_arg_u32(kernelParams, 8);
    uint32_t stride_col_dst = launch_arg_u32(kernelParams, 9);
    struct lanxin_kernel_uint3 channel_ratio = launch_arg_uint3(kernelParams, 10);
    uint32_t stride_channel_x = launch_arg_u32(kernelParams, 11);
    uint32_t stride_channel_y = launch_arg_u32(kernelParams, 12);
    uint32_t stride_channel_dst = launch_arg_u32(kernelParams, 13);
    struct lanxin_kernel_uint3 sample_ratio = launch_arg_uint3(kernelParams, 14);
    uint32_t stride_sample_x = launch_arg_u32(kernelParams, 15);
    uint32_t stride_sample_y = launch_arg_u32(kernelParams, 16);
    uint32_t stride_sample_dst = launch_arg_u32(kernelParams, 17);
    uint32_t ids_stride = launch_arg_u32(kernelParams, 18);
    if (ncols_x % 256 != 0 || stride_col_dst == 0) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    uint32_t blocks_per_row_x = ncols_x / 256U;
    uint32_t y_blocks_per_col = ncols_x / 32U;

    for (uint32_t sample_dst = 0; sample_dst < gridDimZ; sample_dst++) {
        uint32_t sample_x = packed_div_u32(sample_dst, sample_ratio);
        uint32_t sample_y = sample_dst;
        for (uint32_t channel_dst = 0; channel_dst < gridDimY; channel_dst++) {
            uint32_t channel_x;
            uint32_t channel_y;
            if (ids != NULL) {
                uint32_t idx_stride = ids_stride == 0 ? 1U : ids_stride;
                channel_x = (uint32_t)ids[(size_t)channel_dst * idx_stride];
                channel_y = packed_mod_u32(channel_dst, nchannels_y);
            } else {
                channel_x = packed_div_u32(channel_dst, channel_ratio);
                channel_y = channel_dst;
            }
            struct lanxin_block_q8_1 *yb = vy + (size_t)sample_y * stride_sample_y +
                                           (size_t)channel_y * stride_channel_y;
            for (uint32_t row = 0; row < stride_col_dst; row++) {
                size_t x_offset = (size_t)sample_x * stride_sample_x +
                                  (size_t)channel_x * stride_channel_x +
                                  (size_t)row * stride_row_x;
                double acc = 0.0;
                for (uint32_t kbx = 0; kbx < blocks_per_row_x; kbx++) {
                    const struct lanxin_block_q6_K *xb = vx + x_offset + kbx;
                    for (int elem = 0; elem < 256; elem++) {
                        const struct lanxin_block_q8_1 *q8 = yb + (size_t)kbx * 8U + (uint32_t)elem / 32U;
                        float yv = lanxin_half_to_float(q8->d) * (float)q8->qs[elem % 32];
                        acc += (double)q6_k_value(xb, elem) * (double)yv;
                    }
                }
                dst[(size_t)sample_dst * stride_sample_dst +
                    (size_t)channel_dst * stride_channel_dst + row] = (float)acc;
            }
        }
    }
    (void)stride_col_y;
    (void)y_blocks_per_col;
    tracef("host fallback kernel %s rows=%u ncols_x=%u", name, stride_col_dst, ncols_x);
    return CUDA_SUCCESS;
}

static CUresult try_host_kernel_fallback_locked(CUfunction f,
                                                unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                                void **kernelParams)
{
    (void)blockDimY;
    (void)blockDimZ;
    if (f == NULL || f->name == NULL || kernelParams == NULL ||
        env_disabled("LANXIN_NVIDIA_CUDA_HOST_FALLBACK")) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    const char *name = f->name;
    if (strstr(name, "k_compute_batched_ptrs") != NULL) {
        return host_fallback_compute_batched_ptrs_locked(f, gridDimX, gridDimY,
                                                         blockDimX, blockDimY, kernelParams);
    }
    if (strstr(name, "_ZL11k_bin_bcast") != NULL) {
        return host_fallback_bin_bcast_locked(f, gridDimX, gridDimY, gridDimZ, kernelParams);
    }
    if (strstr(name, "_ZL12rms_norm_f32") != NULL) {
        return host_fallback_rms_norm_locked(f, gridDimX, gridDimY, gridDimZ, kernelParams);
    }
    if (strstr(name, "_ZL16k_get_rows_floatIffE") != NULL) {
        return host_fallback_get_rows_float_locked(f, gridDimX, gridDimY, gridDimZ, blockDimX, kernelParams);
    }
    if (strstr(name, "_ZL13quantize_q8_1") != NULL) {
        return host_fallback_quantize_q8_1_locked(f, gridDimY, gridDimZ, kernelParams);
    }
    if (strstr(name, "direct_copy_kernel_cuda") != NULL) {
        CUresult direct_rc = host_fallback_torch_direct_copy_locked(f, kernelParams);
        if (direct_rc == CUDA_SUCCESS) {
            return direct_rc;
        }
        return host_fallback_torch_nocast_direct_copy_locked(f, kernelParams);
    }
    if (strstr(name, "float16_copy_kernel_cuda") != NULL ||
        strstr(name, "bfloat16_copy_kernel_cuda") != NULL) {
        return host_fallback_torch_float_to_lowp_copy_locked(f, kernelParams);
    }
    if (strstr(name, "FillFunctor") != NULL) {
        return host_fallback_torch_fill_locked(f, kernelParams);
    }
    if (strstr(name, "_ZL10rope_multi") != NULL) {
        return host_fallback_rope_multi_float_locked(f, gridDimX, blockDimX, kernelParams);
    }
    if (strstr(name, "_ZL13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0E") != NULL) {
        return host_fallback_mmvq_q6k_ncols1_locked(f, gridDimY, gridDimZ, kernelParams);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuLaunchKernel(CUfunction f,
                                unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void **kernelParams, void **extra)
{
    if (!valid_function(f) || !valid_stream(hStream)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    void *param_copy = NULL;
    size_t param_size = 0;
    rm_u32 param_flags = 0;
    CUresult param_result = capture_launch_params(kernelParams, extra, &param_copy, &param_size, &param_flags);
    if (param_result != CUDA_SUCCESS) {
        return param_result;
    }
    struct launch_request req = {
        .grid = {gridDimX, gridDimY, gridDimZ},
        .block = {blockDimX, blockDimY, blockDimZ},
        .shared_mem_bytes = sharedMemBytes,
        .params = param_copy,
        .params_size = param_size,
        .param_flags = param_flags,
    };
    bool noop_success = env_enabled("LANXIN_NVIDIA_CUDA_NOOP_KERNEL");
    bool rm_submit_only = env_enabled("LANXIN_NVIDIA_CUDA_RM_SUBMIT");
    bool rm_pb_submit = env_enabled("LANXIN_NVIDIA_CUDA_PB_SUBMIT");
    bool rm_qmd_submit = env_enabled("LANXIN_NVIDIA_CUDA_QMD_SUBMIT");
    bool strict_launch = env_enabled("LANXIN_NVIDIA_CUDA_STRICT_LAUNCH");
    bool fail_unsupported_kernel = env_enabled("LANXIN_NVIDIA_CUDA_FAIL_UNSUPPORTED_KERNEL");
    bool default_pb_submit = !noop_success && !rm_submit_only && !env_disabled("LANXIN_NVIDIA_CUDA_PB_SUBMIT");
    pthread_mutex_lock(&g_lock);
    CUresult init = ensure_initialized_locked();
    CUresult host_rc = CUDA_ERROR_NOT_SUPPORTED;
    if (init == CUDA_SUCCESS) {
        host_rc = try_host_kernel_fallback_locked(f, gridDimX, gridDimY, gridDimZ,
                                                  blockDimX, blockDimY, blockDimZ,
                                                  kernelParams);
    }
    int submit_rc = -1;
    if (host_rc == CUDA_SUCCESS) {
        submit_rc = 0;
    } else if (init == CUDA_SUCCESS && fail_unsupported_kernel) {
        submit_rc = -2;
    } else if (init == CUDA_SUCCESS) {
        if (noop_success || rm_submit_only) {
            submit_rc = rm_submit_noop_locked();
        } else if (rm_pb_submit || rm_qmd_submit || default_pb_submit) {
            submit_rc = rm_submit_compute_set_object_locked(f, &req, rm_qmd_submit);
            if (submit_rc != 0 && !rm_qmd_submit && !strict_launch) {
                submit_rc = rm_submit_noop_locked();
            }
        }
    }
    pthread_mutex_unlock(&g_lock);
    tracef("RM cuLaunchKernel scaffold(%s) submit_rc=%d host_rc=%d noop_success=%d submit_only=%d pb_submit=%d qmd_submit=%d default_pb=%d strict=%d fail_unsupported=%d params=%zu flags=0x%x",
           f->name != NULL ? f->name : "<unnamed>", submit_rc,
           (int)host_rc, noop_success ? 1 : 0, rm_submit_only ? 1 : 0, rm_pb_submit ? 1 : 0,
           rm_qmd_submit ? 1 : 0, default_pb_submit ? 1 : 0, strict_launch ? 1 : 0,
           fail_unsupported_kernel ? 1 : 0, param_size, param_flags);
    free(param_copy);
    if (host_rc == CUDA_SUCCESS) {
        return CUDA_SUCCESS;
    }
    if (fail_unsupported_kernel && submit_rc == -2) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (submit_rc != 0) {
        if (noop_success) {
            return CUDA_ERROR_UNKNOWN;
        }
        if (strict_launch || rm_submit_only) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        return CUDA_SUCCESS;
    }
    if (rm_submit_only || (strict_launch && !rm_qmd_submit)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **kernelParams, void **extra)
{
    if (config == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return cuLaunchKernel(f,
                          config->gridDimX, config->gridDimY, config->gridDimZ,
                          config->blockDimX, config->blockDimY, config->blockDimZ,
                          config->sharedMemBytes, config->hStream,
                          kernelParams, extra);
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

CUresult CUDAAPI cuTensorMapEncodeTiled(CUtensorMap *tensorMap,
                                        CUtensorMapDataType tensorDataType,
                                        cuuint32_t tensorRank,
                                        void *globalAddress,
                                        const cuuint64_t *globalDim,
                                        const cuuint64_t *globalStrides,
                                        const cuuint32_t *boxDim,
                                        const cuuint32_t *elementStrides,
                                        CUtensorMapInterleave interleave,
                                        CUtensorMapSwizzle swizzle,
                                        CUtensorMapL2promotion l2Promotion,
                                        CUtensorMapFloatOOBfill oobFill)
{
    (void)tensorDataType;
    (void)tensorRank;
    (void)globalAddress;
    (void)globalDim;
    (void)globalStrides;
    (void)boxDim;
    (void)elementStrides;
    (void)interleave;
    (void)swizzle;
    (void)l2Promotion;
    (void)oobFill;
    if (tensorMap == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    memset(tensorMap, 0, sizeof(*tensorMap));
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuTensorMapEncodeIm2col(CUtensorMap *tensorMap,
                                         CUtensorMapDataType tensorDataType,
                                         cuuint32_t tensorRank,
                                         void *globalAddress,
                                         const cuuint64_t *globalDim,
                                         const cuuint64_t *globalStrides,
                                         const int *pixelBoxLowerCorner,
                                         const int *pixelBoxUpperCorner,
                                         cuuint32_t channelsPerPixel,
                                         cuuint32_t pixelsPerColumn,
                                         const cuuint32_t *elementStrides,
                                         CUtensorMapInterleave interleave,
                                         CUtensorMapSwizzle swizzle,
                                         CUtensorMapL2promotion l2Promotion,
                                         CUtensorMapFloatOOBfill oobFill)
{
    (void)pixelBoxLowerCorner;
    (void)pixelBoxUpperCorner;
    (void)channelsPerPixel;
    (void)pixelsPerColumn;
    return cuTensorMapEncodeTiled(tensorMap, tensorDataType, tensorRank,
                                  globalAddress, globalDim, globalStrides,
                                  NULL, elementStrides, interleave, swizzle,
                                  l2Promotion, oobFill);
}

CUresult CUDAAPI cuTensorMapEncodeIm2colWide(CUtensorMap *tensorMap,
                                             CUtensorMapDataType tensorDataType,
                                             cuuint32_t tensorRank,
                                             void *globalAddress,
                                             const cuuint64_t *globalDim,
                                             const cuuint64_t *globalStrides,
                                             int pixelBoxLowerCornerWidth,
                                             int pixelBoxUpperCornerWidth,
                                             cuuint32_t channelsPerPixel,
                                             cuuint32_t pixelsPerColumn,
                                             const cuuint32_t *elementStrides,
                                             CUtensorMapInterleave interleave,
                                             CUtensorMapIm2ColWideMode mode,
                                             CUtensorMapSwizzle swizzle,
                                             CUtensorMapL2promotion l2Promotion,
                                             CUtensorMapFloatOOBfill oobFill)
{
    (void)pixelBoxLowerCornerWidth;
    (void)pixelBoxUpperCornerWidth;
    (void)channelsPerPixel;
    (void)pixelsPerColumn;
    (void)mode;
    return cuTensorMapEncodeTiled(tensorMap, tensorDataType, tensorRank,
                                  globalAddress, globalDim, globalStrides,
                                  NULL, elementStrides, interleave, swizzle,
                                  l2Promotion, oobFill);
}

CUresult CUDAAPI cuTensorMapReplaceAddress(CUtensorMap *tensorMap, void *globalAddress)
{
    (void)globalAddress;
    if (tensorMap == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
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
    PROC_ENTRY(cuTensorMapEncodeTiled),
    PROC_ENTRY(cuTensorMapEncodeIm2col),
    PROC_ENTRY(cuTensorMapEncodeIm2colWide),
    PROC_ENTRY(cuTensorMapReplaceAddress),
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
