/*
 * NVML stub header for hetGPU RISC-V build
 * Complete stub with all types and declarations needed by PyTorch c10
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===== Return codes ===== */
typedef enum nvmlReturn_enum {
    NVML_SUCCESS = 0,
    NVML_ERROR_UNINITIALIZED = 1,
    NVML_ERROR_INVALID_ARGUMENT = 2,
    NVML_ERROR_NOT_SUPPORTED = 3,
    NVML_ERROR_NO_PERMISSION = 4,
    NVML_ERROR_ALREADY_INITIALIZED = 5,
    NVML_ERROR_NOT_FOUND = 6,
    NVML_ERROR_INSUFFICIENT_SIZE = 7,
    NVML_ERROR_INSUFFICIENT_POWER = 8,
    NVML_ERROR_DRIVER_NOT_LOADED = 9,
    NVML_ERROR_TIMEOUT = 10,
    NVML_ERROR_IRQ_ISSUE = 11,
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,
    NVML_ERROR_FUNCTION_NOT_FOUND = 13,
    NVML_ERROR_CORRUPTED_INFOROM = 14,
    NVML_ERROR_GPU_IS_LOST = 15,
    NVML_ERROR_RESET_REQUIRED = 16,
    NVML_ERROR_OPERATING_SYSTEM = 17,
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18,
    NVML_ERROR_IN_USE = 19,
    NVML_ERROR_MEMORY = 20,
    NVML_ERROR_NO_DATA = 21,
    NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22,
    NVML_ERROR_UNKNOWN = 999,
} nvmlReturn_t;

/* ===== Device handle ===== */
typedef struct nvmlDevice_st* nvmlDevice_t;

/* ===== Enable/disable state ===== */
typedef enum { NVML_FEATURE_DISABLED = 0, NVML_FEATURE_ENABLED = 1 } nvmlEnableState_t;

/* ===== Process info ===== */
typedef struct nvmlProcessInfo_v1_st {
    unsigned int pid;
    unsigned long long usedGpuMemory;
    unsigned int gpuInstanceId;
    unsigned int computeInstanceId;
} nvmlProcessInfo_v1_t;
typedef nvmlProcessInfo_v1_t nvmlProcessInfo_t;

/* ===== PCI info ===== */
typedef struct nvmlPciInfo_st {
    char busIdLegacy[16];
    unsigned int domain;
    unsigned int bus;
    unsigned int device;
    unsigned int pciDeviceId;
    unsigned int pciSubSystemId;
    char busId[32];
} nvmlPciInfo_t;

/* ===== NvLink device type ===== */
typedef enum {
    NVML_NVLINK_DEVICE_TYPE_GPU = 0,
    NVML_NVLINK_DEVICE_TYPE_IBMNPU = 1,
    NVML_NVLINK_DEVICE_TYPE_SWITCH = 2,
    NVML_NVLINK_DEVICE_TYPE_UNKNOWN = 0xFF,
} nvmlIntNvLinkDeviceType_t;

/* ===== GPU Fabric state ===== */
typedef enum {
    NVML_GPU_FABRIC_STATE_NOT_SUPPORTED = 0,
    NVML_GPU_FABRIC_STATE_NOT_STARTED = 1,
    NVML_GPU_FABRIC_STATE_IN_PROGRESS = 2,
    NVML_GPU_FABRIC_STATE_COMPLETED = 3,
} nvmlGpuFabricState_t;

/* ===== GPU Fabric Info (versioned) ===== */
/* Version 2 structure */
typedef struct nvmlGpuFabricInfoV_st {
    unsigned int version;
    unsigned char clusterUuid[16];
    unsigned long long cliqueId;
    nvmlGpuFabricState_t state;
    nvmlReturn_t status;
    unsigned int healthMask;
    /* v3 fields (accessed conditionally via nvmlGpuFabricInfo_v3) */
    unsigned char healthSummary;
    unsigned char reserved[7];
} nvmlGpuFabricInfoV_t;

/* Versioning constants for nvmlGpuFabricInfoV_t */
#define nvmlGpuFabricInfo_v2 0x02000204UL
#define nvmlGpuFabricInfo_v3 0x03000214UL

/* ===== Clock types ===== */
typedef enum {
    NVML_CLOCK_GRAPHICS = 0,
    NVML_CLOCK_SM = 1,
    NVML_CLOCK_MEM = 2,
    NVML_CLOCK_VIDEO = 3,
    NVML_CLOCK_COUNT
} nvmlClockType_t;

/* ===== Memory info ===== */
typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

/* ===== Function declarations ===== */

/* Init/shutdown */
nvmlReturn_t nvmlInit(void);
nvmlReturn_t nvmlInit_v2(void);
nvmlReturn_t nvmlShutdown(void);

/* Device enumeration */
nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount);
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount);

/* Device handle retrieval */
nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device);
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device);
nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId, nvmlDevice_t *device);
nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, nvmlDevice_t *device);

/* Device info */
nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length);
nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci);
nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory);
nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz);
nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t *current, nvmlEnableState_t *pending);
nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length);
nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed);
nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, unsigned int sensorType, unsigned int *temp);

/* Process info */
nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos);
nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos);
nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos);

/* NvLink */
nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t *pNvLinkDeviceType);
nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci);

/* System info */
nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *cudaDriverVersion);

/* GPU Fabric */
nvmlReturn_t nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, nvmlGpuFabricInfoV_t *gpuFabricInfo);

/* Error string */
const char* nvmlErrorString(nvmlReturn_t result);

#ifdef __cplusplus
}
#endif

/* ===== Additional macros ===== */
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE       32
#define NVML_DEVICE_PCI_BUS_ID_FMT               "%08X:%02X:%02X.0"
