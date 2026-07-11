#include "../include/cuda.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define FATBINC_MAGIC 0x466243b1
#define SYNTH_ELF_SIZE 0x1d0u

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

static int fail(CUresult result, const char *what)
{
    const char *name = NULL;
    const char *desc = NULL;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &desc);
    fprintf(stderr, "%s failed: %s (%s)\n", what, name ? name : "?", desc ? desc : "?");
    return 1;
}

static void build_synthetic_elf(unsigned char *image)
{
    memset(image, 0, SYNTH_ELF_SIZE);
    image[0] = 0x7f;
    image[1] = 'E';
    image[2] = 'L';
    image[3] = 'F';
    image[4] = 2; /* ELFCLASS64 */
    image[5] = 1; /* ELFDATA2LSB */
    image[6] = 1; /* EV_CURRENT */
    wr16le(image + 16, 1);    /* ET_REL */
    wr16le(image + 18, 190);  /* EM_CUDA */
    wr32le(image + 20, 1);    /* EV_CURRENT */
    wr64le(image + 40, 0x80); /* e_shoff */
    wr16le(image + 52, 64);   /* e_ehsize */
    wr16le(image + 58, 64);   /* e_shentsize */
    wr16le(image + 60, 2);    /* e_shnum */

    unsigned char *section = image + 0x80 + 64;
    wr32le(section + 4, 1);       /* SHT_PROGBITS */
    wr64le(section + 24, 0x180);  /* sh_offset */
    wr64le(section + 32, 0x50);   /* sh_size */
    memcpy(image + 0x180, "lanxin synthetic cuda code object", 33);
}

int main(void)
{
    static unsigned long long elf_storage[(SYNTH_ELF_SIZE + 7u) / 8u];
    unsigned char *elf_image = (unsigned char *)elf_storage;
    CUmodule mod_elf = NULL;
    CUmodule mod_fatbin = NULL;

    build_synthetic_elf(elf_image);
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

    result = cuModuleLoadFatBinary(&mod_fatbin, &wrapper);
    if (result != CUDA_SUCCESS) return fail(result, "cuModuleLoadFatBinary(wrapper)");

    cuModuleUnload(mod_fatbin);
    cuModuleUnload(mod_elf);

    printf("module_image_probe result=ok expected_size=%u\n", SYNTH_ELF_SIZE);
    return 0;
}
