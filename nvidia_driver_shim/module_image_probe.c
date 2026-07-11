#include "../include/cuda.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FATBINC_MAGIC 0x466243b1
#define SYNTH_ELF_SIZE 0x3c0u
#define SHSTRTAB_OFF 0x40u
#define TEXT_OFF 0x180u
#define INFO_OFF 0x200u
#define CONST0_OFF 0x240u
#define SHDR_OFF 0x280u
#define SHNUM 5u

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
    uint32_t text_name = add_shstr(shstr, &shcur, ".text.fake_kernel");
    uint32_t info_name = add_shstr(shstr, &shcur, ".nv.info.fake_kernel");
    uint32_t const0_name = add_shstr(shstr, &shcur, ".nv.constant0.fake_kernel");

    for (uint32_t i = 0; i < 64; i++) {
        image[TEXT_OFF + i] = (unsigned char)(0xa0u + (i & 0x1fu));
    }

    size_t info_cur = 0;
    write_attr_u32(image + INFO_OFF, &info_cur, 0x2f04, 48);  /* REGCOUNT */
    write_attr_u32(image + INFO_OFF, &info_cur, 0x0504, 256); /* MAX_THREADS */
    write_attr_u32(image + INFO_OFF, &info_cur, 0x0808, 128); /* SMEM_SIZE */
    write_attr_u32(image + INFO_OFF, &info_cur, 0x0a04, 16);  /* LMEM_SIZE */

    for (uint32_t i = 0; i < 32; i++) {
        image[CONST0_OFF + i] = (unsigned char)i;
    }

    unsigned char *sh = image + SHDR_OFF;
    write_shdr(sh + 1 * 64, shstr_name, 3, 0, 0, SHSTRTAB_OFF, shcur, 0, 0, 1, 0);
    write_shdr(sh + 2 * 64, text_name, 1, 0x6, 0, TEXT_OFF, 64, 0, 0, 128, 0);
    write_shdr(sh + 3 * 64, info_name, 0x70000000u, 0, 0, INFO_OFF, info_cur, 0, 2, 4, 0);
    write_shdr(sh + 4 * 64, const0_name, 1, 0x2, 0, CONST0_OFF, 32, 0, 0, 4, 0);
}

int main(void)
{
    static unsigned long long elf_storage[(SYNTH_ELF_SIZE + 7u) / 8u];
    unsigned char *elf_image = (unsigned char *)elf_storage;
    CUmodule mod_elf = NULL;
    CUmodule mod_fatbin = NULL;
    CUfunction fn_elf = NULL;
    CUfunction fn_fatbin = NULL;

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
    result = cuModuleGetFunction(&fn_elf, mod_elf, "fake_kernel");
    if (result != CUDA_SUCCESS) return fail(result, "cuModuleGetFunction(elf)");

    result = cuModuleLoadFatBinary(&mod_fatbin, &wrapper);
    if (result != CUDA_SUCCESS) return fail(result, "cuModuleLoadFatBinary(wrapper)");
    result = cuModuleGetFunction(&fn_fatbin, mod_fatbin, "fake_kernel");
    if (result != CUDA_SUCCESS) return fail(result, "cuModuleGetFunction(fatbin)");

    if (check_attr(fn_elf, CU_FUNC_ATTRIBUTE_NUM_REGS, 48, "NUM_REGS") ||
        check_attr(fn_elf, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 256, "MAX_THREADS") ||
        check_attr(fn_elf, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, 128, "SHARED_SIZE") ||
        check_attr(fn_elf, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, 16, "LOCAL_SIZE") ||
        check_attr(fn_elf, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, 32, "CONST_SIZE")) {
        return 2;
    }

    if (getenv("LANXIN_NVIDIA_CUDA_MODULE_IMAGE_LAUNCH") != NULL) {
        result = cuLaunchKernel(fn_elf, 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL);
        if (result != CUDA_SUCCESS) {
            if (!((getenv("LANXIN_NVIDIA_CUDA_STRICT_LAUNCH") != NULL ||
                   getenv("LANXIN_NVIDIA_CUDA_QMD_SUBMIT") != NULL) &&
                  result == CUDA_ERROR_NOT_SUPPORTED)) {
                return fail(result, "cuLaunchKernel");
            }
        }
    }

    cuModuleUnload(mod_fatbin);
    cuModuleUnload(mod_elf);

    printf("module_image_probe result=ok expected_size=%u kernel=fake_kernel\n", SYNTH_ELF_SIZE);
    return 0;
}
