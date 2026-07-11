#include "../include/cuda.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FATBINC_MAGIC 0x466243b1
#define SYNTH_ELF_SIZE 0x580u
#define SHSTRTAB_OFF 0x40u
#define TEXT0_OFF 0x180u
#define INFO0_OFF 0x200u
#define CONST0_OFF 0x240u
#define TEXT1_OFF 0x280u
#define INFO1_OFF 0x300u
#define CONST1_OFF 0x340u
#define SHDR_OFF 0x380u
#define SHNUM 8u

typedef struct {
    int magic;
    int version;
    const unsigned long long *data;
    void *filename_or_fatbins;
} fatbin_wrapper_t;

static void wr16le(unsigned char *p, uint16_t v)
{
    p[0] = (unsigned char)(v & 0xffu);
    p[1] = (unsigned char)((v >> 8) & 0xffu);
}

static void wr32le(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)(v & 0xffu);
    p[1] = (unsigned char)((v >> 8) & 0xffu);
    p[2] = (unsigned char)((v >> 16) & 0xffu);
    p[3] = (unsigned char)((v >> 24) & 0xffu);
}

static void wr64le(unsigned char *p, uint64_t v)
{
    wr32le(p, (uint32_t)(v & 0xffffffffu));
    wr32le(p + 4, (uint32_t)(v >> 32));
}

static void write_shdr(unsigned char *p, uint32_t name, uint32_t type, uint64_t flags,
                       uint64_t addr, uint64_t offset, uint64_t size,
                       uint32_t link, uint32_t info, uint64_t align, uint64_t entsize)
{
    wr32le(p + 0, name);
    wr32le(p + 4, type);
    wr64le(p + 8, flags);
    wr64le(p + 16, addr);
    wr64le(p + 24, offset);
    wr64le(p + 32, size);
    wr32le(p + 40, link);
    wr32le(p + 44, info);
    wr64le(p + 48, align);
    wr64le(p + 56, entsize);
}

static uint32_t add_shstr(unsigned char *table, size_t *cursor, const char *name)
{
    uint32_t off = (uint32_t)*cursor;
    size_t len = strlen(name) + 1u;
    memcpy(table + *cursor, name, len);
    *cursor += len;
    return off;
}

static void write_attr_u32(unsigned char *data, size_t *cursor, uint16_t attr, uint32_t value)
{
    wr16le(data + *cursor, attr);
    wr16le(data + *cursor + 2, 4);
    wr32le(data + *cursor + 4, value);
    *cursor += 8;
}

static int fail(CUresult result, const char *what)
{
    const char *name = NULL;
    const char *desc = NULL;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &desc);
    fprintf(stderr, "%s failed: %s (%s)\n", what, name ? name : "?", desc ? desc : "?");
    return 1;
}

static int fail_count(const char *what, unsigned int got, unsigned int expected)
{
    fprintf(stderr, "%s mismatch: got=%u expected>=%u\n", what, got, expected);
    return 2;
}

static int check_attr(CUfunction fn, CUfunction_attribute attrib, int expected, const char *name)
{
    int got = -1;
    CUresult result = cuFuncGetAttribute(&got, attrib, fn);
    if (result != CUDA_SUCCESS) {
        return fail(result, name);
    }
    if (got != expected) {
        fprintf(stderr, "%s mismatch: got=%d expected=%d\n", name, got, expected);
        return 1;
    }
    return 0;
}

static void build_synthetic_cubin(unsigned char *image)
{
    memset(image, 0, SYNTH_ELF_SIZE);

    image[0] = 0x7f;
    image[1] = 'E';
    image[2] = 'L';
    image[3] = 'F';
    image[4] = 2; /* ELFCLASS64 */
    image[5] = 1; /* ELFDATA2LSB */
    image[6] = 1; /* EV_CURRENT */
    wr16le(image + 16, 2);       /* ET_EXEC */
    wr16le(image + 18, 190);     /* EM_CUDA */
    wr32le(image + 20, 1);       /* EV_CURRENT */
    wr64le(image + 40, SHDR_OFF);
    wr32le(image + 48, 120);     /* sm_120 */
    wr16le(image + 52, 64);      /* e_ehsize */
    wr16le(image + 58, 64);      /* e_shentsize */
    wr16le(image + 60, SHNUM);   /* e_shnum */
    wr16le(image + 62, 1);       /* e_shstrndx */

    unsigned char *shstr = image + SHSTRTAB_OFF;
    size_t shcur = 1;
    uint32_t shstr_name = add_shstr(shstr, &shcur, ".shstrtab");
    uint32_t text0_name = add_shstr(shstr, &shcur, ".text.fake_kernel");
    uint32_t info0_name = add_shstr(shstr, &shcur, ".nv.info.fake_kernel");
    uint32_t const0_name = add_shstr(shstr, &shcur, ".nv.constant0.fake_kernel");
    uint32_t text1_name = add_shstr(shstr, &shcur, ".text.second_kernel");
    uint32_t info1_name = add_shstr(shstr, &shcur, ".nv.info.second_kernel");
    uint32_t const1_name = add_shstr(shstr, &shcur, ".nv.constant0.second_kernel");

    for (uint32_t i = 0; i < 64; i++) {
        image[TEXT0_OFF + i] = (unsigned char)(0xa0u + (i & 0x1fu));
    }
    for (uint32_t i = 0; i < 48; i++) {
        image[TEXT1_OFF + i] = (unsigned char)(0xc0u + (i & 0x1fu));
    }

    size_t info0_cur = 0;
    write_attr_u32(image + INFO0_OFF, &info0_cur, 0x2f04, 48);  /* REGCOUNT */
    write_attr_u32(image + INFO0_OFF, &info0_cur, 0x0504, 256); /* MAX_THREADS */
    write_attr_u32(image + INFO0_OFF, &info0_cur, 0x0808, 128); /* SMEM_SIZE */
    write_attr_u32(image + INFO0_OFF, &info0_cur, 0x0a04, 16);  /* LMEM_SIZE */

    size_t info1_cur = 0;
    write_attr_u32(image + INFO1_OFF, &info1_cur, 0x2f04, 24);  /* REGCOUNT */
    write_attr_u32(image + INFO1_OFF, &info1_cur, 0x0504, 128); /* MAX_THREADS */
    write_attr_u32(image + INFO1_OFF, &info1_cur, 0x0808, 64);  /* SMEM_SIZE */
    write_attr_u32(image + INFO1_OFF, &info1_cur, 0x0a04, 0);   /* LMEM_SIZE */

    for (uint32_t i = 0; i < 32; i++) {
        image[CONST0_OFF + i] = (unsigned char)i;
    }
    for (uint32_t i = 0; i < 16; i++) {
        image[CONST1_OFF + i] = (unsigned char)(0x80u + i);
    }

    unsigned char *sh = image + SHDR_OFF;
    write_shdr(sh + 1 * 64, shstr_name, 3, 0, 0, SHSTRTAB_OFF, shcur, 0, 0, 1, 0);
    write_shdr(sh + 2 * 64, text0_name, 1, 0x6, 0, TEXT0_OFF, 64, 0, 0, 128, 0);
    write_shdr(sh + 3 * 64, info0_name, 0x70000000u, 0, 0, INFO0_OFF, info0_cur, 0, 2, 4, 0);
    write_shdr(sh + 4 * 64, const0_name, 1, 0x2, 0, CONST0_OFF, 32, 0, 0, 4, 0);
    write_shdr(sh + 5 * 64, text1_name, 1, 0x6, 0, TEXT1_OFF, 48, 0, 0, 128, 0);
    write_shdr(sh + 6 * 64, info1_name, 0x70000000u, 0, 0, INFO1_OFF, info1_cur, 0, 5, 4, 0);
    write_shdr(sh + 7 * 64, const1_name, 1, 0x2, 0, CONST1_OFF, 16, 0, 0, 4, 0);
}

int main(void)
{
    static unsigned long long elf_storage[(SYNTH_ELF_SIZE + 7u) / 8u];
    unsigned char *elf_image = (unsigned char *)elf_storage;
    CUmodule mod_elf = NULL;
    CUmodule mod_fatbin = NULL;
    CUfunction fn_elf = NULL;
    CUfunction fn_fatbin = NULL;
    CUfunction fn_elf_second = NULL;
    CUfunction fn_fatbin_second = NULL;
    CUlibrary lib_elf = NULL;
    CUkernel kernel_fake = NULL;
    CUkernel kernel_second = NULL;
    CUkernel kernels[4] = {0};
    unsigned int function_count_elf = 0;
    unsigned int function_count_fatbin = 0;
    unsigned int kernel_count = 0;

    build_synthetic_cubin(elf_image);
    fatbin_wrapper_t wrapper = {
        .magic = FATBINC_MAGIC,
        .version = 1,
        .data = elf_storage,
        .filename_or_fatbins = NULL,
    };

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) return fail(result, "cuInit");

    result = cuModuleLoadData(&mod_elf, elf_image);
    if (result != CUDA_SUCCESS) return fail(result, "cuModuleLoadData(elf)");
    result = cuModuleGetFunctionCount(&function_count_elf, mod_elf);
    if (result != CUDA_SUCCESS) return fail(result, "cuModuleGetFunctionCount(elf)");
    if (function_count_elf < 2) return fail_count("function_count_elf", function_count_elf, 2);
    result = cuModuleGetFunction(&fn_elf, mod_elf, "fake_kernel");
    if (result != CUDA_SUCCESS) return fail(result, "cuModuleGetFunction(elf)");
    result = cuModuleGetFunction(&fn_elf_second, mod_elf, "second_kernel");
    if (result != CUDA_SUCCESS) return fail(result, "cuModuleGetFunction(elf second)");

    result = cuModuleLoadFatBinary(&mod_fatbin, &wrapper);
    if (result != CUDA_SUCCESS) return fail(result, "cuModuleLoadFatBinary(wrapper)");
    result = cuModuleGetFunctionCount(&function_count_fatbin, mod_fatbin);
    if (result != CUDA_SUCCESS) return fail(result, "cuModuleGetFunctionCount(fatbin)");
    if (function_count_fatbin < 2) return fail_count("function_count_fatbin", function_count_fatbin, 2);
    result = cuModuleGetFunction(&fn_fatbin, mod_fatbin, "fake_kernel");
    if (result != CUDA_SUCCESS) return fail(result, "cuModuleGetFunction(fatbin)");
    result = cuModuleGetFunction(&fn_fatbin_second, mod_fatbin, "second_kernel");
    if (result != CUDA_SUCCESS) return fail(result, "cuModuleGetFunction(fatbin second)");

    if (check_attr(fn_elf, CU_FUNC_ATTRIBUTE_NUM_REGS, 48, "NUM_REGS") ||
        check_attr(fn_elf, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 256, "MAX_THREADS") ||
        check_attr(fn_elf, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, 128, "SHARED_SIZE") ||
        check_attr(fn_elf, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, 16, "LOCAL_SIZE") ||
        check_attr(fn_elf, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, 32, "CONST_SIZE")) {
        return 2;
    }
    if (check_attr(fn_elf_second, CU_FUNC_ATTRIBUTE_NUM_REGS, 24, "SECOND_NUM_REGS") ||
        check_attr(fn_elf_second, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 128, "SECOND_MAX_THREADS") ||
        check_attr(fn_elf_second, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, 64, "SECOND_SHARED_SIZE") ||
        check_attr(fn_elf_second, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, 0, "SECOND_LOCAL_SIZE") ||
        check_attr(fn_elf_second, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, 16, "SECOND_CONST_SIZE")) {
        return 2;
    }

    result = cuLibraryLoadData(&lib_elf, elf_image, NULL, NULL, 0, NULL, NULL, 0);
    if (result != CUDA_SUCCESS) return fail(result, "cuLibraryLoadData(elf)");
    result = cuLibraryGetKernelCount(&kernel_count, lib_elf);
    if (result != CUDA_SUCCESS) return fail(result, "cuLibraryGetKernelCount(elf)");
    if (kernel_count < 2) return fail_count("kernel_count", kernel_count, 2);
    result = cuLibraryEnumerateKernels(kernels, 4, lib_elf);
    if (result != CUDA_SUCCESS) return fail(result, "cuLibraryEnumerateKernels(elf)");
    if (kernels[0] == NULL || kernels[1] == NULL) {
        fprintf(stderr, "library enumerate did not return two kernels\n");
        return 2;
    }
    result = cuLibraryGetKernel(&kernel_fake, lib_elf, "fake_kernel");
    if (result != CUDA_SUCCESS) return fail(result, "cuLibraryGetKernel(fake)");
    result = cuLibraryGetKernel(&kernel_second, lib_elf, "second_kernel");
    if (result != CUDA_SUCCESS) return fail(result, "cuLibraryGetKernel(second)");
    if (kernel_fake == kernel_second) {
        fprintf(stderr, "library kernels alias unexpectedly\n");
        return 2;
    }

    if (getenv("LANXIN_NVIDIA_CUDA_MODULE_IMAGE_LAUNCH") != NULL) {
        CUfunction launch_fn = getenv("LANXIN_NVIDIA_CUDA_MODULE_IMAGE_LAUNCH_SECOND") != NULL ?
                               fn_elf_second : fn_elf;
        result = cuLaunchKernel(launch_fn, 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL);
        if (result != CUDA_SUCCESS) {
            if (!((getenv("LANXIN_NVIDIA_CUDA_STRICT_LAUNCH") != NULL ||
                   getenv("LANXIN_NVIDIA_CUDA_QMD_SUBMIT") != NULL) &&
                  result == CUDA_ERROR_NOT_SUPPORTED)) {
                return fail(result, "cuLaunchKernel");
            }
        }
    }

    cuLibraryUnload(lib_elf);
    cuModuleUnload(mod_fatbin);
    cuModuleUnload(mod_elf);

    printf("module_image_probe result=ok expected_size=%u kernels=%u library_kernels=%u primary=fake_kernel secondary=second_kernel\n",
           SYNTH_ELF_SIZE, function_count_elf, kernel_count);
    return 0;
}
