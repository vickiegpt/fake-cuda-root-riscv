#define _GNU_SOURCE
#include "../include/nvml.h"
#include "../include/cuda.h"

#include <ctype.h>
#include <dirent.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define MAX_GPUS 16U
#define MAX_PROCS 256U
#define MAX_FIELDS 128U

struct proc_row {
    unsigned int pid;
    char name[256];
    unsigned long long used_gpu_memory;
};

struct gpu_row {
    unsigned int index;
    nvmlDevice_t device;
    char name[128];
    char uuid[96];
    char bus_id[32];
    char driver_version[64];
    char cuda_version[32];
    char vbios[64];
    char link_speed[64];
    char link_width[32];
    char max_link_speed[64];
    char max_link_width[32];
    nvmlMemory_t memory;
    int memory_valid;
    int cc_major;
    int cc_minor;
    int sm_count;
    unsigned int graphics_clock_mhz;
    unsigned int memory_clock_mhz;
    int graphics_clock_valid;
    int memory_clock_valid;
    nvmlUtilization_t utilization;
    int utilization_valid;
    unsigned int temperature_c;
    int temperature_valid;
    unsigned int power_mw;
    int power_valid;
    struct proc_row procs[MAX_PROCS];
    unsigned int proc_count;
};

static void trim(char *s)
{
    if (s == NULL) {
        return;
    }
    char *p = s;
    while (*p != '\0' && isspace((unsigned char)*p)) {
        p++;
    }
    if (p != s) {
        memmove(s, p, strlen(p) + 1U);
    }
    size_t n = strlen(s);
    while (n != 0 && isspace((unsigned char)s[n - 1U])) {
        s[--n] = '\0';
    }
}

static int read_first_line(const char *path, char *out, size_t out_len)
{
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return -1;
    }
    if (fgets(out, (int)out_len, f) == NULL) {
        fclose(f);
        return -1;
    }
    fclose(f);
    trim(out);
    return 0;
}

static int read_proc_gpu_field(const char *bus_id, const char *key, char *out, size_t out_len)
{
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "/proc/driver/nvidia/gpus/%s/information", bus_id);
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return -1;
    }
    char line[512];
    size_t key_len = strlen(key);
    int found = -1;
    while (fgets(line, sizeof(line), f) != NULL) {
        trim(line);
        if (strncmp(line, key, key_len) == 0 && line[key_len] == ':') {
            char *p = line + key_len + 1U;
            trim(p);
            size_t n = strnlen(p, out_len - 1U);
            memcpy(out, p, n);
            out[n] = '\0';
            found = 0;
            break;
        }
    }
    fclose(f);
    return found;
}

static int read_sysfs_pci(const char *bus_id, const char *file, char *out, size_t out_len)
{
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/%s", bus_id, file);
    return read_first_line(path, out, out_len);
}

static const char *mib_string(unsigned long long bytes, char *buf, size_t buf_len, int with_unit)
{
    unsigned long long mib = bytes / (1024ULL * 1024ULL);
    snprintf(buf, buf_len, with_unit ? "%llu MiB" : "%llu", mib);
    return buf;
}

static const char *memory_value(unsigned long long bytes, char *buf, size_t buf_len, int nounits)
{
    return mib_string(bytes, buf, buf_len, !nounits);
}

static void cuda_version_string(int driver_version, char *out, size_t out_len)
{
    if (driver_version <= 0) {
        snprintf(out, out_len, "N/A");
        return;
    }
    int major = driver_version / 1000;
    int minor = (driver_version % 1000) / 10;
    snprintf(out, out_len, "%d.%d", major, minor);
}

static void now_string(char *out, size_t out_len)
{
    time_t t = time(NULL);
    struct tm tmv;
    localtime_r(&t, &tmv);
    strftime(out, out_len, "%a %b %d %H:%M:%S %Y", &tmv);
}

static void read_process_name(unsigned int pid, char *out, size_t out_len)
{
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "/proc/%u/comm", pid);
    if (read_first_line(path, out, out_len) == 0 && out[0] != '\0') {
        return;
    }
    snprintf(path, sizeof(path), "/proc/%u/cmdline", pid);
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        snprintf(out, out_len, "pid-%u", pid);
        return;
    }
    size_t n = fread(out, 1, out_len - 1U, f);
    fclose(f);
    if (n == 0) {
        snprintf(out, out_len, "pid-%u", pid);
        return;
    }
    out[n] = '\0';
    for (size_t i = 0; i < n; i++) {
        if (out[i] == '\0') {
            out[i] = ' ';
        }
    }
    trim(out);
}

static int collect_gpu_row(unsigned int index, nvmlDevice_t device, const char *driver_version,
                           const char *cuda_version, struct gpu_row *row)
{
    memset(row, 0, sizeof(*row));
    row->index = index;
    row->device = device;
    snprintf(row->name, sizeof(row->name), "NVIDIA GPU");
    snprintf(row->uuid, sizeof(row->uuid), "N/A");
    snprintf(row->bus_id, sizeof(row->bus_id), "N/A");
    snprintf(row->driver_version, sizeof(row->driver_version), "%s", driver_version);
    snprintf(row->cuda_version, sizeof(row->cuda_version), "%s", cuda_version);
    snprintf(row->vbios, sizeof(row->vbios), "N/A");
    snprintf(row->link_speed, sizeof(row->link_speed), "N/A");
    snprintf(row->link_width, sizeof(row->link_width), "N/A");
    snprintf(row->max_link_speed, sizeof(row->max_link_speed), "N/A");
    snprintf(row->max_link_width, sizeof(row->max_link_width), "N/A");

    if (nvmlDeviceGetName(device, row->name, sizeof(row->name)) != NVML_SUCCESS) {
        return -1;
    }
    (void)nvmlDeviceGetUUID(device, row->uuid, sizeof(row->uuid));
    nvmlPciInfo_t pci;
    if (nvmlDeviceGetPciInfo_v3(device, &pci) == NVML_SUCCESS) {
        snprintf(row->bus_id, sizeof(row->bus_id), "%s", pci.busId);
    }
    if (nvmlDeviceGetMemoryInfo(device, &row->memory) == NVML_SUCCESS) {
        row->memory_valid = 1;
    }
    (void)read_proc_gpu_field(row->bus_id, "Video BIOS", row->vbios, sizeof(row->vbios));
    (void)read_sysfs_pci(row->bus_id, "current_link_speed", row->link_speed, sizeof(row->link_speed));
    (void)read_sysfs_pci(row->bus_id, "current_link_width", row->link_width, sizeof(row->link_width));
    (void)read_sysfs_pci(row->bus_id, "max_link_speed", row->max_link_speed, sizeof(row->max_link_speed));
    (void)read_sysfs_pci(row->bus_id, "max_link_width", row->max_link_width, sizeof(row->max_link_width));

    row->temperature_valid =
        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &row->temperature_c) == NVML_SUCCESS;
    row->power_valid = nvmlDeviceGetPowerUsage(device, &row->power_mw) == NVML_SUCCESS;
    row->utilization_valid = nvmlDeviceGetUtilizationRates(device, &row->utilization) == NVML_SUCCESS;
    row->graphics_clock_valid =
        nvmlDeviceGetClock(device, NVML_CLOCK_GRAPHICS, &row->graphics_clock_mhz) == NVML_SUCCESS;
    row->memory_clock_valid =
        nvmlDeviceGetClock(device, NVML_CLOCK_MEM, &row->memory_clock_mhz) == NVML_SUCCESS;

    CUdevice cu_dev = 0;
    if (cuDeviceGet(&cu_dev, (int)index) == CUDA_SUCCESS) {
        (void)cuDeviceGetAttribute(&row->cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_dev);
        (void)cuDeviceGetAttribute(&row->cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_dev);
        (void)cuDeviceGetAttribute(&row->sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cu_dev);
    }

    unsigned int proc_count = 0;
    nvmlReturn_t rc = nvmlDeviceGetComputeRunningProcesses(device, &proc_count, NULL);
    if (rc == NVML_SUCCESS && proc_count != 0) {
        if (proc_count > MAX_PROCS) {
            proc_count = MAX_PROCS;
        }
        nvmlProcessInfo_t infos[MAX_PROCS];
        memset(infos, 0, sizeof(infos));
        unsigned int cap = proc_count;
        rc = nvmlDeviceGetComputeRunningProcesses(device, &cap, infos);
        if (rc == NVML_SUCCESS || rc == NVML_ERROR_INSUFFICIENT_SIZE) {
            row->proc_count = cap < MAX_PROCS ? cap : MAX_PROCS;
            for (unsigned int i = 0; i < row->proc_count; i++) {
                row->procs[i].pid = infos[i].pid;
                row->procs[i].used_gpu_memory = infos[i].usedGpuMemory;
                read_process_name(infos[i].pid, row->procs[i].name, sizeof(row->procs[i].name));
            }
        }
    }
    return 0;
}

static int collect_gpus(struct gpu_row *rows, unsigned int *row_count)
{
    nvmlReturn_t rc = nvmlInit_v2();
    if (rc != NVML_SUCCESS) {
        fprintf(stderr, "nvidia-smi: nvmlInit_v2 failed: %s\n", nvmlErrorString(rc));
        return 1;
    }
    char driver_version[64] = "N/A";
    if (nvmlSystemGetDriverVersion(driver_version, sizeof(driver_version)) != NVML_SUCCESS) {
        snprintf(driver_version, sizeof(driver_version), "N/A");
    }
    int cuda_driver_version = 0;
    if (nvmlSystemGetCudaDriverVersion_v2(&cuda_driver_version) != NVML_SUCCESS) {
        cuda_driver_version = 0;
    }
    char cuda_version[32];
    cuda_version_string(cuda_driver_version, cuda_version, sizeof(cuda_version));

    unsigned int count = 0;
    rc = nvmlDeviceGetCount_v2(&count);
    if (rc != NVML_SUCCESS) {
        fprintf(stderr, "nvidia-smi: nvmlDeviceGetCount failed: %s\n", nvmlErrorString(rc));
        return 1;
    }
    if (count > MAX_GPUS) {
        count = MAX_GPUS;
    }
    unsigned int out = 0;
    for (unsigned int i = 0; i < count; i++) {
        nvmlDevice_t device = NULL;
        if (nvmlDeviceGetHandleByIndex_v2(i, &device) != NVML_SUCCESS) {
            continue;
        }
        if (collect_gpu_row(i, device, driver_version, cuda_version, &rows[out]) == 0) {
            out++;
        }
    }
    *row_count = out;
    return 0;
}

static void print_help(void)
{
    puts("NVIDIA System Management Interface shim for Lanxin RISC-V");
    puts("");
    puts("Supported options:");
    puts("  nvidia-smi");
    puts("  nvidia-smi -L | --list-gpus");
    puts("  nvidia-smi --query-gpu=field,... --format=csv[,noheader][,nounits]");
    puts("  nvidia-smi --query-compute-apps=field,... --format=csv[,noheader][,nounits]");
    puts("  nvidia-smi --version");
    puts("  nvidia-smi pmon [-c 1]");
    puts("  nvidia-smi dmon [-c 1]");
    puts("");
    puts("Common fields: index,name,uuid,pci.bus_id,driver_version,cuda_version,");
    puts("memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,");
    puts("temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,");
    puts("compute_cap,sm_count,pcie.link.width.current,pcie.link.speed.current");
}

static void print_version(const struct gpu_row *rows, unsigned int count)
{
    const char *driver = count != 0 ? rows[0].driver_version : "N/A";
    const char *cuda = count != 0 ? rows[0].cuda_version : "N/A";
    printf("NVIDIA-SMI %s    Driver Version: %s    CUDA Version: %s\n", driver, driver, cuda);
}

static void print_list_gpus(const struct gpu_row *rows, unsigned int count)
{
    for (unsigned int i = 0; i < count; i++) {
        printf("GPU %u: %s (UUID: %s)\n", rows[i].index, rows[i].name, rows[i].uuid);
    }
}

static void print_default_table(const struct gpu_row *rows, unsigned int count)
{
    char now[64];
    now_string(now, sizeof(now));
    const char *driver = count != 0 ? rows[0].driver_version : "N/A";
    const char *cuda = count != 0 ? rows[0].cuda_version : "N/A";
    printf("%s\n", now);
    puts("+-----------------------------------------------------------------------------------------+");
    printf("| NVIDIA-SMI %-15s Driver Version: %-12s CUDA Version: %-10s |\n", driver, driver, cuda);
    puts("|-----------------------------------------+------------------------+----------------------+");
    puts("| GPU  Name                    Bus-Id     | Memory-Usage           | Volatile Uncorr. ECC |");
    puts("| Fan  Temp   Perf  Pwr:Usage/Cap         | GPU-Util  Compute M.   | MIG M.               |");
    puts("|=========================================+========================+======================|");
    for (unsigned int i = 0; i < count; i++) {
        const struct gpu_row *r = &rows[i];
        char used[32] = "N/A";
        char total[32] = "N/A";
        if (r->memory_valid) {
            (void)mib_string(r->memory.used, used, sizeof(used), 0);
            (void)mib_string(r->memory.total, total, sizeof(total), 0);
        }
        char temp[32];
        snprintf(temp, sizeof(temp), r->temperature_valid ? "%uC" : "N/A", r->temperature_c);
        char power[32];
        if (r->power_valid) {
            snprintf(power, sizeof(power), "%.1fW", (double)r->power_mw / 1000.0);
        } else {
            snprintf(power, sizeof(power), "N/A");
        }
        char util[32];
        snprintf(util, sizeof(util), r->utilization_valid ? "%u%%" : "N/A", r->utilization.gpu);
        printf("| %3u  %-22.22s %-12.12s | %sMiB / %sMiB | N/A                  |\n",
               r->index, r->name, r->bus_id, used, total);
        printf("| N/A  %-5s  P0    %-18s | %-8s Default      | N/A                  |\n",
               temp, power, util);
        puts("+-----------------------------------------+------------------------+----------------------+");
    }
    puts("");
    puts("+-----------------------------------------------------------------------------------------+");
    puts("| Processes:                                                                              |");
    puts("|  GPU   PID      Type   Process name                                           GPU Memory |");
    puts("|=========================================================================================|");
    int any = 0;
    for (unsigned int i = 0; i < count; i++) {
        const struct gpu_row *r = &rows[i];
        for (unsigned int j = 0; j < r->proc_count; j++) {
            const struct proc_row *p = &r->procs[j];
            char mem_buf[32];
            const char *mem = p->used_gpu_memory == NVML_VALUE_NOT_AVAILABLE ?
                "N/A" : mib_string(p->used_gpu_memory, mem_buf, sizeof(mem_buf), 1);
            printf("|  %3u   %-8u C      %-52.52s %10s |\n", r->index, p->pid, p->name, mem);
            any = 1;
        }
    }
    if (!any) {
        puts("|  No running processes found                                                             |");
    }
    puts("+-----------------------------------------------------------------------------------------+");
}

static void csv_cell(const char *value, int last)
{
    int quote = value != NULL && (strchr(value, ',') != NULL || strchr(value, '"') != NULL);
    if (quote) {
        putchar('"');
        for (const char *p = value; *p != '\0'; p++) {
            if (*p == '"') {
                putchar('"');
            }
            putchar(*p);
        }
        putchar('"');
    } else {
        fputs(value != NULL ? value : "", stdout);
    }
    if (!last) {
        fputs(", ", stdout);
    }
}

static const char *gpu_field_value(const struct gpu_row *r, const char *field, int nounits,
                                   char *buf, size_t buf_len)
{
    if (strcmp(field, "timestamp") == 0) {
        now_string(buf, buf_len);
    } else if (strcmp(field, "index") == 0) {
        snprintf(buf, buf_len, "%u", r->index);
    } else if (strcmp(field, "name") == 0 || strcmp(field, "gpu_name") == 0) {
        snprintf(buf, buf_len, "%s", r->name);
    } else if (strcmp(field, "uuid") == 0 || strcmp(field, "gpu_uuid") == 0) {
        snprintf(buf, buf_len, "%s", r->uuid);
    } else if (strcmp(field, "pci.bus_id") == 0 || strcmp(field, "gpu_bus_id") == 0) {
        snprintf(buf, buf_len, "%s", r->bus_id);
    } else if (strcmp(field, "driver_version") == 0) {
        snprintf(buf, buf_len, "%s", r->driver_version);
    } else if (strcmp(field, "cuda_version") == 0) {
        snprintf(buf, buf_len, "%s", r->cuda_version);
    } else if (strcmp(field, "vbios_version") == 0) {
        snprintf(buf, buf_len, "%s", r->vbios);
    } else if (strcmp(field, "memory.total") == 0) {
        if (r->memory_valid) {
            (void)memory_value(r->memory.total, buf, buf_len, nounits);
        } else {
            snprintf(buf, buf_len, "N/A");
        }
    } else if (strcmp(field, "memory.used") == 0) {
        if (r->memory_valid) {
            (void)memory_value(r->memory.used, buf, buf_len, nounits);
        } else {
            snprintf(buf, buf_len, "N/A");
        }
    } else if (strcmp(field, "memory.free") == 0) {
        if (r->memory_valid) {
            (void)memory_value(r->memory.free, buf, buf_len, nounits);
        } else {
            snprintf(buf, buf_len, "N/A");
        }
    } else if (strcmp(field, "utilization.gpu") == 0) {
        snprintf(buf, buf_len, r->utilization_valid ? (nounits ? "%u" : "%u %%") : "N/A", r->utilization.gpu);
    } else if (strcmp(field, "utilization.memory") == 0) {
        snprintf(buf, buf_len, r->utilization_valid ? (nounits ? "%u" : "%u %%") : "N/A", r->utilization.memory);
    } else if (strcmp(field, "temperature.gpu") == 0) {
        snprintf(buf, buf_len, r->temperature_valid ? (nounits ? "%u" : "%u C") : "N/A", r->temperature_c);
    } else if (strcmp(field, "power.draw") == 0) {
        if (r->power_valid) {
            snprintf(buf, buf_len, nounits ? "%.2f" : "%.2f W", (double)r->power_mw / 1000.0);
        } else {
            snprintf(buf, buf_len, "N/A");
        }
    } else if (strcmp(field, "clocks.current.graphics") == 0 || strcmp(field, "clocks.gr") == 0) {
        snprintf(buf, buf_len, r->graphics_clock_valid ? (nounits ? "%u" : "%u MHz") : "N/A",
                 r->graphics_clock_mhz);
    } else if (strcmp(field, "clocks.current.memory") == 0 || strcmp(field, "clocks.mem") == 0) {
        snprintf(buf, buf_len, r->memory_clock_valid ? (nounits ? "%u" : "%u MHz") : "N/A",
                 r->memory_clock_mhz);
    } else if (strcmp(field, "compute_cap") == 0) {
        snprintf(buf, buf_len, "%d.%d", r->cc_major, r->cc_minor);
    } else if (strcmp(field, "sm_count") == 0 || strcmp(field, "multiprocessor_count") == 0) {
        snprintf(buf, buf_len, "%d", r->sm_count);
    } else if (strcmp(field, "pcie.link.width.current") == 0) {
        snprintf(buf, buf_len, "%s", r->link_width);
    } else if (strcmp(field, "pcie.link.speed.current") == 0) {
        snprintf(buf, buf_len, "%s", r->link_speed);
    } else if (strcmp(field, "pcie.link.width.max") == 0) {
        snprintf(buf, buf_len, "%s", r->max_link_width);
    } else if (strcmp(field, "pcie.link.speed.max") == 0) {
        snprintf(buf, buf_len, "%s", r->max_link_speed);
    } else if (strcmp(field, "compute_mode") == 0) {
        snprintf(buf, buf_len, "Default");
    } else if (strcmp(field, "persistence_mode") == 0) {
        snprintf(buf, buf_len, "Disabled");
    } else if (strcmp(field, "display_active") == 0 || strcmp(field, "display_mode") == 0) {
        snprintf(buf, buf_len, "Disabled");
    } else {
        snprintf(buf, buf_len, "N/A");
    }
    return buf;
}

static void print_gpu_query(const struct gpu_row *rows, unsigned int count,
                            char **fields, unsigned int field_count, int noheader, int nounits)
{
    if (!noheader) {
        for (unsigned int i = 0; i < field_count; i++) {
            csv_cell(fields[i], i + 1U == field_count);
        }
        putchar('\n');
    }
    for (unsigned int r = 0; r < count; r++) {
        for (unsigned int f = 0; f < field_count; f++) {
            char value[256];
            csv_cell(gpu_field_value(&rows[r], fields[f], nounits, value, sizeof(value)),
                     f + 1U == field_count);
        }
        putchar('\n');
    }
}

static const char *process_field_value(const struct gpu_row *gpu, const struct proc_row *proc,
                                       const char *field, int nounits, char *buf, size_t buf_len)
{
    if (strcmp(field, "pid") == 0) {
        snprintf(buf, buf_len, "%u", proc->pid);
    } else if (strcmp(field, "process_name") == 0 || strcmp(field, "name") == 0) {
        snprintf(buf, buf_len, "%s", proc->name);
    } else if (strcmp(field, "used_gpu_memory") == 0) {
        if (proc->used_gpu_memory == NVML_VALUE_NOT_AVAILABLE) {
            snprintf(buf, buf_len, "N/A");
        } else {
            (void)memory_value(proc->used_gpu_memory, buf, buf_len, nounits);
        }
    } else if (strcmp(field, "gpu_uuid") == 0) {
        snprintf(buf, buf_len, "%s", gpu->uuid);
    } else if (strcmp(field, "gpu_bus_id") == 0) {
        snprintf(buf, buf_len, "%s", gpu->bus_id);
    } else {
        snprintf(buf, buf_len, "N/A");
    }
    return buf;
}

static void print_process_query(const struct gpu_row *rows, unsigned int count,
                                char **fields, unsigned int field_count, int noheader, int nounits)
{
    if (!noheader) {
        for (unsigned int i = 0; i < field_count; i++) {
            csv_cell(fields[i], i + 1U == field_count);
        }
        putchar('\n');
    }
    for (unsigned int r = 0; r < count; r++) {
        for (unsigned int p = 0; p < rows[r].proc_count; p++) {
            for (unsigned int f = 0; f < field_count; f++) {
                char value[256];
                csv_cell(process_field_value(&rows[r], &rows[r].procs[p], fields[f], nounits,
                                             value, sizeof(value)),
                         f + 1U == field_count);
            }
            putchar('\n');
        }
    }
}

static unsigned int split_fields(char *list, char **fields, unsigned int max_fields)
{
    unsigned int n = 0;
    char *save = NULL;
    for (char *tok = strtok_r(list, ",", &save); tok != NULL && n < max_fields;
         tok = strtok_r(NULL, ",", &save)) {
        trim(tok);
        if (tok[0] != '\0') {
            fields[n++] = tok;
        }
    }
    return n;
}

static void print_query(const struct gpu_row *rows, unsigned int count, const char *query,
                        int apps, const char *format)
{
    char field_buf[2048];
    snprintf(field_buf, sizeof(field_buf), "%s", query);
    char *fields[MAX_FIELDS];
    unsigned int field_count = split_fields(field_buf, fields, MAX_FIELDS);
    if (field_count == 0) {
        return;
    }
    int noheader = format != NULL && strstr(format, "noheader") != NULL;
    int nounits = format != NULL && strstr(format, "nounits") != NULL;
    if (apps) {
        print_process_query(rows, count, fields, field_count, noheader, nounits);
    } else {
        print_gpu_query(rows, count, fields, field_count, noheader, nounits);
    }
}

static void print_pmon(const struct gpu_row *rows, unsigned int count)
{
    puts("# gpu        pid  type    sm   mem   enc   dec   command");
    int any = 0;
    for (unsigned int r = 0; r < count; r++) {
        for (unsigned int p = 0; p < rows[r].proc_count; p++) {
            printf("%4u %10u     C     -     -     -     -   %s\n",
                   rows[r].index, rows[r].procs[p].pid, rows[r].procs[p].name);
            any = 1;
        }
    }
    if (!any) {
        puts("# no running GPU processes found");
    }
}

static void print_dmon(const struct gpu_row *rows, unsigned int count)
{
    puts("# gpu   pwr gtemp mtemp    sm   mem   enc   dec");
    for (unsigned int r = 0; r < count; r++) {
        printf("%4u %5s %5s %5s %5s %5s %5s %5s\n",
               rows[r].index,
               rows[r].power_valid ? "ok" : "-",
               rows[r].temperature_valid ? "ok" : "-",
               "-",
               rows[r].utilization_valid ? "ok" : "-",
               rows[r].utilization_valid ? "ok" : "-",
               "-",
               "-");
    }
}

static int filter_by_id(struct gpu_row *rows, unsigned int *count, const char *id_text)
{
    if (id_text == NULL || id_text[0] == '\0') {
        return 0;
    }
    char *end = NULL;
    unsigned long wanted = strtoul(id_text, &end, 10);
    if (end == id_text || *end != '\0') {
        return 0;
    }
    for (unsigned int i = 0; i < *count; i++) {
        if (rows[i].index == (unsigned int)wanted) {
            if (i != 0) {
                rows[0] = rows[i];
            }
            *count = 1;
            return 0;
        }
    }
    *count = 0;
    return 0;
}

int main(int argc, char **argv)
{
    const char *query_gpu = NULL;
    const char *query_apps = NULL;
    const char *format = "csv";
    const char *id_filter = NULL;
    int list_gpus = 0;
    int show_help = 0;
    int show_version = 0;
    int show_pmon = 0;
    int show_dmon = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            show_help = 1;
        } else if (strcmp(argv[i], "-L") == 0 || strcmp(argv[i], "--list-gpus") == 0) {
            list_gpus = 1;
        } else if (strcmp(argv[i], "--version") == 0) {
            show_version = 1;
        } else if (strncmp(argv[i], "--query-gpu=", 12) == 0) {
            query_gpu = argv[i] + 12;
        } else if (strcmp(argv[i], "--query-gpu") == 0 && i + 1 < argc) {
            query_gpu = argv[++i];
        } else if (strncmp(argv[i], "--query-compute-apps=", 21) == 0) {
            query_apps = argv[i] + 21;
        } else if (strcmp(argv[i], "--query-compute-apps") == 0 && i + 1 < argc) {
            query_apps = argv[++i];
        } else if (strncmp(argv[i], "--format=", 9) == 0) {
            format = argv[i] + 9;
        } else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            format = argv[++i];
        } else if (strncmp(argv[i], "--id=", 5) == 0) {
            id_filter = argv[i] + 5;
        } else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--id") == 0) && i + 1 < argc) {
            id_filter = argv[++i];
        } else if (strcmp(argv[i], "pmon") == 0) {
            show_pmon = 1;
        } else if (strcmp(argv[i], "dmon") == 0) {
            show_dmon = 1;
        }
    }

    if (show_help) {
        print_help();
        return 0;
    }

    struct gpu_row rows[MAX_GPUS];
    unsigned int count = 0;
    int rc = collect_gpus(rows, &count);
    if (rc != 0) {
        return rc;
    }
    (void)filter_by_id(rows, &count, id_filter);

    if (show_version) {
        print_version(rows, count);
    } else if (list_gpus) {
        print_list_gpus(rows, count);
    } else if (query_gpu != NULL) {
        print_query(rows, count, query_gpu, 0, format);
    } else if (query_apps != NULL) {
        print_query(rows, count, query_apps, 1, format);
    } else if (show_pmon) {
        print_pmon(rows, count);
    } else if (show_dmon) {
        print_dmon(rows, count);
    } else {
        print_default_table(rows, count);
    }

    (void)nvmlShutdown();
    return 0;
}
