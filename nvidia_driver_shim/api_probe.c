#include "../include/cuda.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static const char ptx_image[] =
    ".version 8.0\n"
    ".target sm_90\n"
    ".address_size 64\n"
    ".visible .global .align 4 .u32 fake_global;\n"
    ".visible .entry fake_kernel(.param .u64 arg0) {\n"
    "    ret;\n"
    "}\n";

static void print_result(const char *what, CUresult result)
{
    const char *name = NULL;
    const char *desc = NULL;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &desc);
    fprintf(stderr, "%s: %s (%s)\n", what, name ? name : "?", desc ? desc : "?");
}

static void check_ok(CUresult result, const char *what)
{
    if (result != CUDA_SUCCESS) {
        print_result(what, result);
        failures++;
    }
}

static void check_true(int condition, const char *what)
{
    if (!condition) {
        fprintf(stderr, "%s: failed\n", what);
        failures++;
    }
}

static void check_proc(const char *name)
{
    void *fn = NULL;
    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    CUresult result = cuGetProcAddress(name, &fn, 12090, 0, &status);
    if (result != CUDA_SUCCESS || fn == NULL || status != CU_GET_PROC_ADDRESS_SUCCESS) {
        fprintf(stderr, "cuGetProcAddress(%s) failed result=%d status=%d fn=%p\n",
                name, result, status, fn);
        failures++;
    }
}

static void write_probe_file(const char *path)
{
    FILE *fp = fopen(path, "wb");
    check_true(fp != NULL, "fopen probe file");
    if (fp == NULL) {
        return;
    }
    check_true(fwrite(ptx_image, 1, sizeof(ptx_image), fp) == sizeof(ptx_image),
               "write probe file");
    fclose(fp);
}

static void test_memory(CUstream stream)
{
    CUdeviceptr d0 = 0;
    CUdeviceptr d1 = 0;
    CUdeviceptr managed = 0;
    CUdeviceptr pitched = 0;
    size_t pitch = 0;
    unsigned char in[256];
    unsigned char out[256];
    memset(out, 0, sizeof(out));
    for (size_t i = 0; i < sizeof(in); i++) {
        in[i] = (unsigned char)(i ^ 0xa5U);
    }

    check_ok(cuMemAlloc(&d0, sizeof(in)), "cuMemAlloc");
    check_ok(cuMemAllocManaged(&managed, 128, 0), "cuMemAllocManaged");
    check_ok(cuMemAllocPitch(&pitched, &pitch, 33, 4, 1), "cuMemAllocPitch");
    check_true(pitch >= 33, "pitch >= width");
    check_ok(cuMemAlloc(&d1, sizeof(in)), "cuMemAlloc(second)");
    check_ok(cuMemcpyHtoD(d0, in, sizeof(in)), "cuMemcpyHtoD");
    check_ok(cuMemcpyDtoD(d1, d0, sizeof(in)), "cuMemcpyDtoD");
    check_ok(cuMemcpyDtoH(out, d1, sizeof(out)), "cuMemcpyDtoH");
    check_true(memcmp(in, out, sizeof(in)) == 0, "DtoD roundtrip");

    memset(out, 0, sizeof(out));
    check_ok(cuMemcpyHtoDAsync(d0, in, sizeof(in), stream), "cuMemcpyHtoDAsync");
    check_ok(cuMemcpyDtoHAsync(out, d0, sizeof(out), stream), "cuMemcpyDtoHAsync");
    check_ok(cuStreamSynchronize(stream), "cuStreamSynchronize(after async copy)");
    check_true(memcmp(in, out, sizeof(in)) == 0, "async roundtrip");

    check_ok(cuMemsetD8(d0, 0x5a, 64), "cuMemsetD8");
    memset(out, 0, sizeof(out));
    check_ok(cuMemcpyDtoH(out, d0, 64), "cuMemcpyDtoH(after memset8)");
    check_true(out[0] == 0x5a && out[63] == 0x5a, "memset8 data");
    check_ok(cuMemsetD16(d0, 0x1234, 8), "cuMemsetD16");
    check_ok(cuMemsetD32(d0, 0x89abcdefU, 4), "cuMemsetD32");
    check_ok(cuMemsetD8Async(d0, 0x11, 16, stream), "cuMemsetD8Async");
    check_ok(cuMemsetD16Async(d0, 0x2222, 4, stream), "cuMemsetD16Async");
    check_ok(cuMemsetD32Async(d0, 0x33333333U, 2, stream), "cuMemsetD32Async");

    CUDA_MEMCPY2D copy;
    memset(&copy, 0, sizeof(copy));
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = in;
    copy.srcPitch = 16;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = d0;
    copy.dstPitch = 16;
    copy.WidthInBytes = 16;
    copy.Height = 4;
    check_ok(cuMemcpy2D(&copy), "cuMemcpy2D HtoD");
    memset(out, 0, sizeof(out));
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = d0;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = out;
    check_ok(cuMemcpy2DAsync(&copy, stream), "cuMemcpy2DAsync DtoH");
    check_true(memcmp(in, out, 64) == 0, "2D roundtrip");

    unsigned int mem_type = 0;
    CUdeviceptr range_base = 0;
    size_t range_size = 0;
    check_ok(cuPointerGetAttribute(&mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, d0),
             "cuPointerGetAttribute(memory_type)");
    check_ok(cuMemGetAddressRange(&range_base, &range_size, d0), "cuMemGetAddressRange");
    check_true(range_base == d0 && range_size >= sizeof(in), "address range");
    CUpointer_attribute attrs[3] = {
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        CU_POINTER_ATTRIBUTE_RANGE_SIZE,
    };
    void *values[3] = {&mem_type, &range_base, &range_size};
    check_ok(cuPointerGetAttributes(3, attrs, values, d0), "cuPointerGetAttributes");

    void *host = NULL;
    CUdeviceptr host_dev = 0;
    check_ok(cuMemAllocHost(&host, 128), "cuMemAllocHost");
    check_ok(cuMemHostGetDevicePointer(&host_dev, host, 0), "cuMemHostGetDevicePointer");
    check_true(host_dev == (CUdeviceptr)(uintptr_t)host, "host device pointer");
    check_ok(cuMemFreeHost(host), "cuMemFreeHost");

    void *registered = malloc(128);
    check_true(registered != NULL, "malloc registered");
    if (registered != NULL) {
        check_ok(cuMemHostRegister(registered, 128, 0), "cuMemHostRegister");
        check_ok(cuMemHostGetDevicePointer(&host_dev, registered, 0),
                 "cuMemHostGetDevicePointer(registered)");
        check_ok(cuMemHostUnregister(registered), "cuMemHostUnregister");
        free(registered);
    }

    check_ok(cuMemFree(d1), "cuMemFree(second)");
    check_ok(cuMemFree(d0), "cuMemFree");
    check_ok(cuMemFree(managed), "cuMemFree(managed)");
    check_ok(cuMemFree(pitched), "cuMemFree(pitched)");
}

static void test_module_and_launch(const char *path)
{
    CUmodule mod = NULL;
    CUfunction fn = NULL;
    CUmodule mod_file = NULL;
    CUmodule mod_fat = NULL;
    CUdeviceptr global = 0;
    size_t global_bytes = 0;
    unsigned int function_count = 0;
    const char *fn_name = NULL;
    CUmodule fn_module = NULL;
    int attr = 0;
    int occ = 0;
    int min_grid = 0;
    int block_size = 0;
    size_t dyn_smem = 0;

    check_ok(cuModuleLoadDataEx(&mod, ptx_image, 0, NULL, NULL), "cuModuleLoadDataEx");
    check_ok(cuModuleGetFunction(&fn, mod, "fake_kernel"), "cuModuleGetFunction");
    check_ok(cuModuleGetFunctionCount(&function_count, mod), "cuModuleGetFunctionCount");
    check_true(function_count >= 1, "function count");
    check_ok(cuFuncGetName(&fn_name, fn), "cuFuncGetName");
    check_true(fn_name != NULL && strcmp(fn_name, "fake_kernel") == 0, "function name");
    check_ok(cuFuncGetModule(&fn_module, fn), "cuFuncGetModule");
    check_true(fn_module == mod, "function module");
    check_ok(cuFuncGetAttribute(&attr, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn),
             "cuFuncGetAttribute");
    check_ok(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0),
             "cuFuncSetAttribute");
    check_ok(cuFuncSetCacheConfig(fn, CU_FUNC_CACHE_PREFER_NONE), "cuFuncSetCacheConfig");
    check_ok(cuFuncSetSharedMemConfig(fn, CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE),
             "cuFuncSetSharedMemConfig");
    check_ok(cuModuleGetGlobal(&global, &global_bytes, mod, "fake_global"),
             "cuModuleGetGlobal");
    check_true(global != 0 && global_bytes > 0, "module global");
    check_ok(cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, 128, 0),
             "cuOccupancyMaxActiveBlocksPerMultiprocessor");
    check_ok(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occ, fn, 128, 0, 0),
             "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    check_ok(cuOccupancyMaxPotentialBlockSize(&min_grid, &block_size, fn, NULL, 0, 0),
             "cuOccupancyMaxPotentialBlockSize");
    check_ok(cuOccupancyMaxPotentialBlockSizeWithFlags(&min_grid, &block_size, fn, NULL, 0, 0, 0),
             "cuOccupancyMaxPotentialBlockSizeWithFlags");
    check_ok(cuOccupancyAvailableDynamicSMemPerBlock(&dyn_smem, fn, 1, 128),
             "cuOccupancyAvailableDynamicSMemPerBlock");
    uint64_t arg0 = 0x1122334455667788ULL;
    size_t arg_size = sizeof(arg0);
    void *launch_extra[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, &arg0,
        CU_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        CU_LAUNCH_PARAM_END,
    };
    check_ok(cuLaunchKernel(fn, 2, 1, 1, 32, 1, 1, 16, NULL, NULL, launch_extra),
             "cuLaunchKernel");
    CUlaunchConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.gridDimX = 1;
    cfg.gridDimY = 1;
    cfg.gridDimZ = 1;
    cfg.blockDimX = 1;
    cfg.blockDimY = 1;
    cfg.blockDimZ = 1;
    check_ok(cuLaunchKernelEx(&cfg, fn, NULL, NULL), "cuLaunchKernelEx");

    check_ok(cuModuleLoad(&mod_file, path), "cuModuleLoad");
    check_ok(cuModuleLoadFatBinary(&mod_fat, ptx_image), "cuModuleLoadFatBinary");
    check_ok(cuModuleUnload(mod_fat), "cuModuleUnload(fat)");
    check_ok(cuModuleUnload(mod_file), "cuModuleUnload(file)");
    check_ok(cuModuleUnload(mod), "cuModuleUnload");
}

static void test_link_and_library(const char *path)
{
    CUlinkState link = NULL;
    void *cubin = NULL;
    size_t cubin_size = 0;
    CUmodule linked_mod = NULL;
    CUlibrary lib = NULL;
    CUlibrary lib_file = NULL;
    CUkernel kernel = NULL;
    CUkernel kernels[4] = {0};
    unsigned int kernel_count = 0;
    CUfunction fn = NULL;
    CUlibrary kernel_lib = NULL;
    CUmodule lib_mod = NULL;
    CUdeviceptr global = 0;
    size_t global_bytes = 0;
    void *unified = NULL;
    int attr = 0;
    const char *kernel_name = NULL;
    size_t param_offset = 0;
    size_t param_size = 0;

    check_ok(cuLinkCreate(0, NULL, NULL, &link), "cuLinkCreate");
    check_ok(cuLinkAddData(link, CU_JIT_INPUT_PTX, (void *)ptx_image, sizeof(ptx_image),
                           "inline.ptx", 0, NULL, NULL), "cuLinkAddData");
    check_ok(cuLinkAddFile(link, CU_JIT_INPUT_PTX, path, 0, NULL, NULL), "cuLinkAddFile");
    check_ok(cuLinkComplete(link, &cubin, &cubin_size), "cuLinkComplete");
    check_true(cubin != NULL && cubin_size > 0, "link output");
    check_ok(cuModuleLoadData(&linked_mod, cubin), "cuModuleLoadData(linked)");
    check_ok(cuModuleUnload(linked_mod), "cuModuleUnload(linked)");
    check_ok(cuLinkDestroy(link), "cuLinkDestroy");

    check_ok(cuLibraryLoadData(&lib, ptx_image, NULL, NULL, 0, NULL, NULL, 0),
             "cuLibraryLoadData");
    check_ok(cuLibraryGetKernel(&kernel, lib, "fake_kernel"), "cuLibraryGetKernel");
    check_ok(cuLibraryGetKernelCount(&kernel_count, lib), "cuLibraryGetKernelCount");
    check_true(kernel_count >= 1, "library kernel count");
    check_ok(cuLibraryEnumerateKernels(kernels, 4, lib), "cuLibraryEnumerateKernels");
    check_true(kernels[0] != NULL, "library enumerate");
    check_ok(cuKernelGetFunction(&fn, kernel), "cuKernelGetFunction");
    check_ok(cuKernelGetLibrary(&kernel_lib, kernel), "cuKernelGetLibrary");
    check_true(kernel_lib == lib, "kernel library");
    check_ok(cuLibraryGetModule(&lib_mod, lib), "cuLibraryGetModule");
    check_ok(cuLibraryGetGlobal(&global, &global_bytes, lib, "fake_global"),
             "cuLibraryGetGlobal");
    check_ok(cuLibraryGetManaged(&global, &global_bytes, lib, "fake_managed"),
             "cuLibraryGetManaged");
    check_ok(cuLibraryGetUnifiedFunction(&unified, lib, "fake_kernel"),
             "cuLibraryGetUnifiedFunction");
    check_true(unified != NULL, "unified function");
    check_ok(cuKernelGetAttribute(&attr, CU_FUNC_ATTRIBUTE_BINARY_VERSION, kernel, 0),
             "cuKernelGetAttribute");
    check_ok(cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0, kernel, 0),
             "cuKernelSetAttribute");
    check_ok(cuKernelSetCacheConfig(kernel, CU_FUNC_CACHE_PREFER_NONE, 0),
             "cuKernelSetCacheConfig");
    check_ok(cuKernelGetName(&kernel_name, kernel), "cuKernelGetName");
    check_true(kernel_name != NULL && strcmp(kernel_name, "fake_kernel") == 0, "kernel name");
    check_ok(cuKernelGetParamInfo(kernel, 0, &param_offset, &param_size),
             "cuKernelGetParamInfo");
    check_true(param_size == sizeof(void *), "kernel param size");
    check_ok(cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL),
             "cuLaunchKernel(library)");
    check_ok(cuLibraryUnload(lib), "cuLibraryUnload");

    check_ok(cuLibraryLoadFromFile(&lib_file, path, NULL, NULL, 0, NULL, NULL, 0),
             "cuLibraryLoadFromFile");
    check_ok(cuLibraryUnload(lib_file), "cuLibraryUnload(file)");
}

static void test_proc_table(void)
{
    static const char *names[] = {
        "cuInit", "cuDriverGetVersion", "cuDeviceGetCount", "cuDeviceGet",
        "cuDeviceGetName", "cuDeviceGetUuid", "cuDeviceGetUuid_v2",
        "cuDeviceTotalMem", "cuDeviceTotalMem_v2", "cuDeviceGetAttribute",
        "cuDeviceComputeCapability", "cuDeviceGetPCIBusId", "cuDeviceGetByPCIBusId",
        "cuCtxCreate", "cuCtxCreate_v2", "cuCtxCreate_v3", "cuCtxDestroy",
        "cuCtxDestroy_v2", "cuCtxSetCurrent", "cuCtxGetCurrent", "cuCtxPushCurrent",
        "cuCtxPopCurrent", "cuCtxGetDevice", "cuCtxGetFlags", "cuCtxSynchronize",
        "cuCtxGetApiVersion", "cuCtxGetId", "cuMemGetInfo", "cuMemGetInfo_v2",
        "cuMemAlloc", "cuMemAlloc_v2", "cuMemAllocManaged", "cuMemAllocPitch",
        "cuMemAllocPitch_v2", "cuMemFree", "cuMemFree_v2", "cuMemGetAddressRange",
        "cuMemGetAddressRange_v2", "cuMemAllocHost", "cuMemAllocHost_v2",
        "cuMemFreeHost", "cuMemHostAlloc", "cuMemHostRegister", "cuMemHostUnregister",
        "cuMemHostGetDevicePointer", "cuMemHostGetDevicePointer_v2", "cuMemcpy",
        "cuMemcpyAsync", "cuMemcpyHtoD", "cuMemcpyHtoD_v2", "cuMemcpyDtoH",
        "cuMemcpyDtoH_v2", "cuMemcpyDtoD", "cuMemcpyDtoD_v2", "cuMemcpyHtoDAsync",
        "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoHAsync", "cuMemcpyDtoHAsync_v2",
        "cuMemcpyDtoDAsync", "cuMemcpyDtoDAsync_v2", "cuMemcpy2D", "cuMemcpy2D_v2",
        "cuMemcpy2DUnaligned", "cuMemcpy2DUnaligned_v2", "cuMemcpy2DAsync",
        "cuMemcpy2DAsync_v2", "cuMemsetD8", "cuMemsetD8_v2", "cuMemsetD16",
        "cuMemsetD16_v2", "cuMemsetD32", "cuMemsetD32_v2", "cuMemsetD8Async",
        "cuMemsetD16Async", "cuMemsetD32Async", "cuPointerGetAttribute",
        "cuPointerGetAttributes", "cuStreamCreate", "cuStreamCreateWithPriority",
        "cuStreamDestroy", "cuStreamDestroy_v2", "cuStreamQuery", "cuStreamSynchronize",
        "cuStreamGetPriority", "cuStreamGetFlags", "cuStreamGetId", "cuStreamGetCtx",
        "cuStreamGetDevice", "cuEventCreate", "cuEventRecord", "cuEventRecordWithFlags",
        "cuEventQuery", "cuEventSynchronize", "cuEventDestroy", "cuEventDestroy_v2",
        "cuEventElapsedTime", "cuEventElapsedTime_v2", "cuModuleLoad", "cuModuleLoadData",
        "cuModuleLoadDataEx", "cuModuleLoadFatBinary", "cuModuleGetGlobal",
        "cuModuleGetGlobal_v2", "cuModuleUnload", "cuModuleGetFunction",
        "cuModuleGetFunctionCount", "cuLinkCreate", "cuLinkCreate_v2", "cuLinkAddData",
        "cuLinkAddData_v2", "cuLinkAddFile", "cuLinkAddFile_v2", "cuLinkComplete",
        "cuLinkDestroy", "cuFuncGetName", "cuFuncGetModule", "cuFuncGetAttribute",
        "cuFuncSetAttribute", "cuFuncSetCacheConfig", "cuFuncSetSharedMemConfig",
        "cuLibraryLoadData", "cuLibraryLoadFromFile", "cuLibraryUnload",
        "cuLibraryGetKernel", "cuLibraryGetKernelCount", "cuLibraryEnumerateKernels",
        "cuLibraryGetModule", "cuKernelGetFunction", "cuKernelGetLibrary",
        "cuLibraryGetGlobal", "cuLibraryGetManaged", "cuLibraryGetUnifiedFunction",
        "cuKernelGetAttribute", "cuKernelSetAttribute", "cuKernelSetCacheConfig",
        "cuKernelGetName", "cuKernelGetParamInfo", "cuLaunchKernel",
        "cuLaunchKernelEx", "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        "cuOccupancyMaxPotentialBlockSize", "cuOccupancyMaxPotentialBlockSizeWithFlags",
        "cuOccupancyAvailableDynamicSMemPerBlock", "cuGetErrorName", "cuGetErrorString",
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        check_proc(names[i]);
    }
}

int main(void)
{
    const char *path = "/tmp/lanxin_fake_cuda_api_probe.ptx";
    int version = 0;
    int count = 0;
    CUdevice dev = 0;
    CUdevice by_pci = -1;
    CUuuid uuid;
    char name[128] = {0};
    char pci[64] = {0};
    size_t total = 0;
    size_t free_bytes = 0;
    int attr = 0;
    int cc_major = 0;
    int cc_minor = 0;
    CUcontext ctx = NULL;
    CUcontext popped = NULL;
    CUcontext current = NULL;
    unsigned int flags = 0;
    unsigned int api_version = 0;
    unsigned long long ctx_id = 0;
    CUstream stream = NULL;
    CUstream priority_stream = NULL;
    int priority = 0;
    unsigned long long stream_id = 0;
    CUevent start = NULL;
    CUevent stop = NULL;
    float elapsed_ms = 0.0f;

    write_probe_file(path);
    check_ok(cuInit(0), "cuInit");
    check_ok(cuDriverGetVersion(&version), "cuDriverGetVersion");
    check_ok(cuDeviceGetCount(&count), "cuDeviceGetCount");
    check_true(count > 0, "device count");
    check_ok(cuDeviceGet(&dev, 0), "cuDeviceGet");
    check_ok(cuDeviceGetName(name, sizeof(name), dev), "cuDeviceGetName");
    check_ok(cuDeviceGetUuid(&uuid, dev), "cuDeviceGetUuid");
    check_ok(cuDeviceGetUuid_v2(&uuid, dev), "cuDeviceGetUuid_v2");
    check_ok(cuDeviceTotalMem(&total, dev), "cuDeviceTotalMem");
    check_ok(cuDeviceTotalMem_v2(&total, dev), "cuDeviceTotalMem_v2");
    check_ok(cuDeviceComputeCapability(&cc_major, &cc_minor, dev), "cuDeviceComputeCapability");
    check_ok(cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev),
             "cuDeviceGetAttribute");
    check_ok(cuDeviceGetPCIBusId(pci, sizeof(pci), dev), "cuDeviceGetPCIBusId");
    check_ok(cuDeviceGetByPCIBusId(&by_pci, pci), "cuDeviceGetByPCIBusId");
    check_true(by_pci == dev, "device by pci");

    check_ok(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");
    check_ok(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
    check_true(current == ctx, "current context");
    check_ok(cuCtxGetDevice(&dev), "cuCtxGetDevice");
    check_ok(cuCtxGetFlags(&flags), "cuCtxGetFlags");
    check_ok(cuCtxGetApiVersion(ctx, &api_version), "cuCtxGetApiVersion");
    check_ok(cuCtxGetId(ctx, &ctx_id), "cuCtxGetId");
    check_ok(cuCtxSynchronize(), "cuCtxSynchronize");
    check_ok(cuMemGetInfo(&free_bytes, &total), "cuMemGetInfo");

    check_ok(cuStreamCreate(&stream, 0), "cuStreamCreate");
    check_ok(cuStreamCreateWithPriority(&priority_stream, 0, -1), "cuStreamCreateWithPriority");
    check_ok(cuStreamGetPriority(priority_stream, &priority), "cuStreamGetPriority");
    check_ok(cuStreamGetFlags(priority_stream, &flags), "cuStreamGetFlags");
    check_ok(cuStreamGetId(priority_stream, &stream_id), "cuStreamGetId");
    check_ok(cuStreamGetCtx(priority_stream, &current), "cuStreamGetCtx");
    check_ok(cuStreamGetDevice(priority_stream, &dev), "cuStreamGetDevice");
    check_ok(cuStreamQuery(priority_stream), "cuStreamQuery");

    check_ok(cuEventCreate(&start, 0), "cuEventCreate(start)");
    check_ok(cuEventCreate(&stop, 0), "cuEventCreate(stop)");
    check_ok(cuEventRecord(start, stream), "cuEventRecord");
    check_ok(cuEventRecordWithFlags(stop, stream, 0), "cuEventRecordWithFlags");
    check_ok(cuEventQuery(stop), "cuEventQuery");
    check_ok(cuEventSynchronize(stop), "cuEventSynchronize");
    check_ok(cuEventElapsedTime(&elapsed_ms, start, stop), "cuEventElapsedTime");
    check_ok(cuEventElapsedTime_v2(&elapsed_ms, start, stop), "cuEventElapsedTime_v2");

    test_memory(stream);
    test_module_and_launch(path);
    test_link_and_library(path);
    test_proc_table();

    check_ok(cuEventDestroy(stop), "cuEventDestroy(stop)");
    check_ok(cuEventDestroy_v2(start), "cuEventDestroy_v2(start)");
    check_ok(cuStreamDestroy(priority_stream), "cuStreamDestroy(priority)");
    check_ok(cuStreamDestroy_v2(stream), "cuStreamDestroy_v2(stream)");
    check_ok(cuCtxPushCurrent(ctx), "cuCtxPushCurrent");
    check_ok(cuCtxPopCurrent(&popped), "cuCtxPopCurrent");
    check_true(popped == ctx, "popped context");
    check_ok(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
    check_ok(cuCtxDestroy(ctx), "cuCtxDestroy");
    remove(path);

    printf("api_probe result=%s failures=%d device=\"%s\" pci=%s cc=%d.%d total=%zu free=%zu elapsed_ms=%.3f\n",
           failures == 0 ? "ok" : "fail", failures, name, pci,
           cc_major, cc_minor, total, free_bytes, elapsed_ms);
    return failures == 0 ? 0 : 1;
}
