#include "../include/cuda.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>

static int check(CUresult result, const char *what)
{
    if (result == CUDA_SUCCESS) {
        return 0;
    }
    const char *name = NULL;
    const char *desc = NULL;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &desc);
    fprintf(stderr, "%s failed: %s (%s)\n", what, name ? name : "?", desc ? desc : "?");
    return 1;
}

int main(void)
{
    int failures = 0;
    int version = 0;
    int count = 0;
    CUdevice dev = 0;
    char name[128] = {0};
    char pci[64] = {0};
    size_t total = 0;
    size_t free_bytes = 0;
    int cc_major = 0;
    int cc_minor = 0;
    int sm_count = 0;
    CUcontext ctx = NULL;
    CUdeviceptr dptr = 0;
    unsigned char in[4096];
    unsigned char out[4096];

    failures += check(cuInit(0), "cuInit");
    failures += check(cuDriverGetVersion(&version), "cuDriverGetVersion");
    failures += check(cuDeviceGetCount(&count), "cuDeviceGetCount");
    if (count <= 0) {
        fprintf(stderr, "no devices reported\n");
        return 2;
    }
    failures += check(cuDeviceGet(&dev, 0), "cuDeviceGet");
    failures += check(cuDeviceGetName(name, sizeof(name), dev), "cuDeviceGetName");
    failures += check(cuDeviceGetPCIBusId(pci, sizeof(pci), dev), "cuDeviceGetPCIBusId");
    failures += check(cuDeviceTotalMem(&total, dev), "cuDeviceTotalMem");
    failures += check(cuDeviceComputeCapability(&cc_major, &cc_minor, dev), "cuDeviceComputeCapability");
    failures += check(cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), "cuDeviceGetAttribute(SM)");
    failures += check(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");
    failures += check(cuMemGetInfo(&free_bytes, &total), "cuMemGetInfo(before)");
    failures += check(cuMemAlloc(&dptr, sizeof(in)), "cuMemAlloc");

    for (size_t i = 0; i < sizeof(in); i++) {
        in[i] = (unsigned char)((i * 131U + 7U) & 0xffU);
        out[i] = 0;
    }
    failures += check(cuMemcpyHtoD(dptr, in, sizeof(in)), "cuMemcpyHtoD");
    failures += check(cuMemcpyDtoH(out, dptr, sizeof(out)), "cuMemcpyDtoH");
    if (memcmp(in, out, sizeof(in)) != 0) {
        fprintf(stderr, "roundtrip data mismatch\n");
        failures++;
    }

    unsigned int memory_type = 0;
    size_t range_size = 0;
    failures += check(cuPointerGetAttribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dptr), "cuPointerGetAttribute(type)");
    failures += check(cuPointerGetAttribute(&range_size, CU_POINTER_ATTRIBUTE_RANGE_SIZE, dptr), "cuPointerGetAttribute(size)");
    failures += check(cuMemFree(dptr), "cuMemFree");
    failures += check(cuCtxDestroy(ctx), "cuCtxDestroy");

    void *fn = NULL;
    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    failures += check(cuGetProcAddress("cuMemAlloc", &fn, version, 0, &status), "cuGetProcAddress(cuMemAlloc)");
    if (fn == NULL || status != CU_GET_PROC_ADDRESS_SUCCESS) {
        fprintf(stderr, "cuGetProcAddress did not return a valid cuMemAlloc pointer\n");
        failures++;
    }

    printf("driver=%d devices=%d name=\"%s\" pci=%s cc=%d.%d sm=%d total=%zu free_before=%zu mem_type=%u range=%zu roundtrip=%s proc=%p\n",
           version, count, name, pci, cc_major, cc_minor, sm_count, total, free_bytes,
           memory_type, range_size, failures == 0 ? "ok" : "fail", fn);

    return failures == 0 ? 0 : 1;
}

