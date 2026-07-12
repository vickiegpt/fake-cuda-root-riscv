#define _GNU_SOURCE
#include "../include/nvml.h"
#include "../include/cuda.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define MAX_NVML_DEVICES 16U
#define MAX_NVML_PROCS 256U
#define MAX_UTIL_TRACKED_PROCS 512U
#define DEFAULT_ACCOUNTING_DIR "/tmp/lanxin_nvidia_cuda_accounting"

struct nvmlDevice_st {
    unsigned int index;
};

static int g_initialized;
static unsigned int g_device_count;
static struct nvmlDevice_st g_devices[MAX_NVML_DEVICES];

typedef struct {
    unsigned int pid;
    unsigned long long timeStamp;
    unsigned int smUtil;
    unsigned int memUtil;
    unsigned int encUtil;
    unsigned int decUtil;
} nvmlProcessUtilizationSample_t;

struct util_sample {
    pid_t pid;
    unsigned long long cpu_ticks;
    double uptime_seconds;
    unsigned int last_util;
};

static struct util_sample g_util_samples[MAX_UTIL_TRACKED_PROCS];

static int process_has_nvidia_fd(pid_t pid);
static int process_is_monitor(pid_t pid);

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

static void copy_string(char *dst, unsigned int dst_len, const char *src)
{
    if (dst == NULL || dst_len == 0) {
        return;
    }
    if (src == NULL) {
        src = "";
    }
    strncpy(dst, src, dst_len - 1U);
    dst[dst_len - 1U] = '\0';
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

static int read_hex_file(const char *path, unsigned int *value)
{
    char line[64];
    if (read_first_line(path, line, sizeof(line)) != 0) {
        return -1;
    }
    char *end = NULL;
    unsigned long parsed = strtoul(line, &end, 0);
    if (end == line) {
        return -1;
    }
    *value = (unsigned int)parsed;
    return 0;
}

static double read_uptime_seconds(void)
{
    FILE *f = fopen("/proc/uptime", "r");
    if (f == NULL) {
        return 0.0;
    }
    double uptime = 0.0;
    if (fscanf(f, "%lf", &uptime) != 1) {
        uptime = 0.0;
    }
    fclose(f);
    return uptime;
}

static int read_process_cpu_ticks(pid_t pid, unsigned long long *cpu_ticks, double *age_seconds)
{
    if (cpu_ticks == NULL || age_seconds == NULL) {
        return -1;
    }

    char path[PATH_MAX];
    snprintf(path, sizeof(path), "/proc/%ld/stat", (long)pid);
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return -1;
    }
    char line[4096];
    if (fgets(line, sizeof(line), f) == NULL) {
        fclose(f);
        return -1;
    }
    fclose(f);

    char *rp = strrchr(line, ')');
    if (rp == NULL || rp[1] == '\0') {
        return -1;
    }

    unsigned long long utime = 0;
    unsigned long long stime = 0;
    unsigned long long starttime = 0;
    unsigned int field = 3;
    char *save = NULL;
    for (char *tok = strtok_r(rp + 2, " ", &save); tok != NULL; tok = strtok_r(NULL, " ", &save), field++) {
        if (field == 14) {
            utime = strtoull(tok, NULL, 10);
        } else if (field == 15) {
            stime = strtoull(tok, NULL, 10);
        } else if (field == 22) {
            starttime = strtoull(tok, NULL, 10);
            break;
        }
    }
    if (utime == 0 && stime == 0 && starttime == 0) {
        return -1;
    }

    long hz = sysconf(_SC_CLK_TCK);
    if (hz <= 0) {
        hz = 100;
    }
    double uptime = read_uptime_seconds();
    double process_start = (double)starttime / (double)hz;
    *cpu_ticks = utime + stime;
    *age_seconds = uptime > process_start ? uptime - process_start : 0.0;
    return 0;
}

static unsigned int clamp_percent(double value)
{
    if (value <= 0.0) {
        return 0;
    }
    if (value >= 100.0) {
        return 100;
    }
    return (unsigned int)(value + 0.5);
}

static unsigned int estimate_process_gpu_util(pid_t pid)
{
    unsigned long long cpu_ticks = 0;
    double age_seconds = 0.0;
    if (read_process_cpu_ticks(pid, &cpu_ticks, &age_seconds) != 0) {
        return 0;
    }

    long hz = sysconf(_SC_CLK_TCK);
    if (hz <= 0) {
        hz = 100;
    }
    double uptime = read_uptime_seconds();
    struct util_sample *slot = NULL;
    struct util_sample *empty = NULL;
    for (unsigned int i = 0; i < MAX_UTIL_TRACKED_PROCS; i++) {
        if (g_util_samples[i].pid == pid) {
            slot = &g_util_samples[i];
            break;
        }
        if (empty == NULL && g_util_samples[i].pid == 0) {
            empty = &g_util_samples[i];
        }
    }
    if (slot == NULL) {
        slot = empty != NULL ? empty : &g_util_samples[(unsigned int)pid % MAX_UTIL_TRACKED_PROCS];
        memset(slot, 0, sizeof(*slot));
        slot->pid = pid;
    }

    double util = 0.0;
    double elapsed = uptime - slot->uptime_seconds;
    if (slot->uptime_seconds > 0.0 && elapsed >= 0.05 && cpu_ticks >= slot->cpu_ticks) {
        util = ((double)(cpu_ticks - slot->cpu_ticks) / (double)hz) * 100.0 / elapsed;
    } else if (slot->uptime_seconds > 0.0 && elapsed < 0.05) {
        util = (double)slot->last_util;
    } else if (age_seconds > 0.0) {
        util = ((double)cpu_ticks / (double)hz) * 100.0 / age_seconds;
    }

    slot->cpu_ticks = cpu_ticks;
    slot->uptime_seconds = uptime;
    slot->last_util = clamp_percent(util);
    return slot->last_util;
}

static unsigned int scan_total_process_gpu_util(void)
{
    DIR *dir = opendir("/proc");
    if (dir == NULL) {
        return 0;
    }

    unsigned int total = 0;
    pid_t self = getpid();
    struct dirent *de = NULL;
    while ((de = readdir(dir)) != NULL) {
        if (!isdigit((unsigned char)de->d_name[0])) {
            continue;
        }
        char *end = NULL;
        long parsed = strtol(de->d_name, &end, 10);
        if (end == de->d_name || *end != '\0' || parsed <= 0 || parsed == (long)self) {
            continue;
        }
        pid_t pid = (pid_t)parsed;
        if (process_is_monitor(pid) || !process_has_nvidia_fd(pid)) {
            continue;
        }
        total += estimate_process_gpu_util(pid);
        if (total >= 100) {
            total = 100;
            break;
        }
    }
    closedir(dir);
    return total;
}

static unsigned long long now_timestamp_us(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return (unsigned long long)time(NULL) * 1000000ULL;
    }
    return (unsigned long long)ts.tv_sec * 1000000ULL + (unsigned long long)ts.tv_nsec / 1000ULL;
}

static const char *accounting_dir(void)
{
    const char *dir = getenv("LANXIN_NVIDIA_CUDA_ACCOUNTING_DIR");
    return (dir != NULL && dir[0] != '\0') ? dir : DEFAULT_ACCOUNTING_DIR;
}

static int pid_is_alive(pid_t pid)
{
    char proc_path[PATH_MAX];
    snprintf(proc_path, sizeof(proc_path), "/proc/%ld", (long)pid);
    return access(proc_path, F_OK) == 0;
}

static int read_process_accounting_file(pid_t pid, unsigned long long *bytes)
{
    if (bytes == NULL || !pid_is_alive(pid)) {
        return -1;
    }

    char path[PATH_MAX];
    int written = snprintf(path, sizeof(path), "%s/%ld.mem", accounting_dir(), (long)pid);
    if (written <= 0 || (size_t)written >= sizeof(path)) {
        return -1;
    }

    char line[64];
    if (read_first_line(path, line, sizeof(line)) != 0) {
        return -1;
    }

    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(line, &end, 10);
    if (errno != 0 || end == line) {
        return -1;
    }
    *bytes = parsed;
    return 0;
}

static unsigned long long read_process_nvidia_mmaps(pid_t pid)
{
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "/proc/%ld/maps", (long)pid);
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return 0;
    }

    unsigned long long bytes = 0;
    char line[1024];
    while (fgets(line, sizeof(line), f) != NULL) {
        if (strstr(line, "/dev/nvidia") == NULL) {
            continue;
        }
        unsigned long long start = 0;
        unsigned long long end = 0;
        if (sscanf(line, "%llx-%llx", &start, &end) == 2 && end > start) {
            bytes += end - start;
        }
    }
    fclose(f);
    return bytes;
}

static unsigned long long read_process_gpu_memory(pid_t pid)
{
    unsigned long long bytes = 0;
    if (read_process_accounting_file(pid, &bytes) == 0) {
        return bytes;
    }

    bytes = read_process_nvidia_mmaps(pid);
    return bytes != 0 ? bytes : NVML_VALUE_NOT_AVAILABLE;
}

static int parse_pci_bus_id(const char *bus_id, unsigned int *domain,
                            unsigned int *bus, unsigned int *device, unsigned int *function)
{
    if (bus_id == NULL) {
        return -1;
    }
    unsigned int d = 0, b = 0, dev = 0, fn = 0;
    if (sscanf(bus_id, "%x:%x:%x.%x", &d, &b, &dev, &fn) == 4) {
        if (domain != NULL) {
            *domain = d;
        }
        if (bus != NULL) {
            *bus = b;
        }
        if (device != NULL) {
            *device = dev;
        }
        if (function != NULL) {
            *function = fn;
        }
        return 0;
    }
    if (sscanf(bus_id, "%x:%x.%x", &b, &dev, &fn) == 3) {
        if (domain != NULL) {
            *domain = 0;
        }
        if (bus != NULL) {
            *bus = b;
        }
        if (device != NULL) {
            *device = dev;
        }
        if (function != NULL) {
            *function = fn;
        }
        return 0;
    }
    return -1;
}

static int cuda_device_for_handle(nvmlDevice_t device, CUdevice *cu_dev)
{
    if (!g_initialized) {
        return NVML_ERROR_UNINITIALIZED;
    }
    if (device == NULL || device < g_devices ||
        device >= g_devices + MAX_NVML_DEVICES || device->index >= g_device_count) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    CUdevice dev = 0;
    CUresult rc = cuDeviceGet(&dev, (int)device->index);
    if (rc != CUDA_SUCCESS) {
        return NVML_ERROR_UNKNOWN;
    }
    if (cu_dev != NULL) {
        *cu_dev = dev;
    }
    return NVML_SUCCESS;
}

static int get_bus_id_for_index(unsigned int index, char *bus_id, size_t bus_id_len)
{
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, (int)index) != CUDA_SUCCESS) {
        return -1;
    }
    if (cuDeviceGetPCIBusId(bus_id, (int)bus_id_len, dev) != CUDA_SUCCESS) {
        return -1;
    }
    return 0;
}

static int read_proc_gpu_field(const char *bus_id, const char *key, char *out, size_t out_len)
{
    if (out == NULL || out_len == 0 || key == NULL) {
        return -1;
    }
    char path[PATH_MAX];
    if (bus_id != NULL && bus_id[0] != '\0') {
        snprintf(path, sizeof(path), "/proc/driver/nvidia/gpus/%s/information", bus_id);
    } else {
        DIR *dir = opendir("/proc/driver/nvidia/gpus");
        if (dir == NULL) {
            return -1;
        }
        struct dirent *de = NULL;
        path[0] = '\0';
        while ((de = readdir(dir)) != NULL) {
            if (de->d_name[0] == '.') {
                continue;
            }
            snprintf(path, sizeof(path), "/proc/driver/nvidia/gpus/%s/information", de->d_name);
            break;
        }
        closedir(dir);
        if (path[0] == '\0') {
            return -1;
        }
    }

    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return -1;
    }
    char line[512];
    int found = -1;
    size_t key_len = strlen(key);
    while (fgets(line, sizeof(line), f) != NULL) {
        char *p = line;
        trim(p);
        if (strncmp(p, key, key_len) == 0 && p[key_len] == ':') {
            p += key_len + 1U;
            trim(p);
            copy_string(out, (unsigned int)out_len, p);
            found = 0;
            break;
        }
    }
    fclose(f);
    return found;
}

static void format_uuid_from_cuda(const CUuuid *uuid, char *out, size_t out_len)
{
    const unsigned char *b = (const unsigned char *)uuid->bytes;
    snprintf(out, out_len,
             "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
             b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
}

static int extract_driver_version(char *out, size_t out_len)
{
    char line[512];
    if (read_first_line("/proc/driver/nvidia/version", line, sizeof(line)) != 0) {
        return -1;
    }
    char copy[512];
    snprintf(copy, sizeof(copy), "%s", line);
    char *save = NULL;
    for (char *tok = strtok_r(copy, " \t", &save); tok != NULL; tok = strtok_r(NULL, " \t", &save)) {
        int has_digit = 0;
        int has_dot = 0;
        int all_version_chars = 1;
        for (char *p = tok; *p != '\0'; p++) {
            if (isdigit((unsigned char)*p)) {
                has_digit = 1;
            } else if (*p == '.') {
                has_dot = 1;
            } else {
                all_version_chars = 0;
                break;
            }
        }
        if (has_digit && has_dot && all_version_chars) {
            snprintf(out, out_len, "%s", tok);
            return 0;
        }
    }
    return -1;
}

static int process_has_nvidia_fd(pid_t pid)
{
    char fd_dir[PATH_MAX];
    snprintf(fd_dir, sizeof(fd_dir), "/proc/%ld/fd", (long)pid);
    DIR *dir = opendir(fd_dir);
    if (dir == NULL) {
        return 0;
    }
    struct dirent *de = NULL;
    int found = 0;
    while ((de = readdir(dir)) != NULL) {
        if (de->d_name[0] == '.') {
            continue;
        }
        char fd_path[PATH_MAX];
        char link[PATH_MAX];
        size_t dir_len = strlen(fd_dir);
        size_t name_len = strlen(de->d_name);
        if (dir_len + 1U + name_len >= sizeof(fd_path)) {
            continue;
        }
        memcpy(fd_path, fd_dir, dir_len);
        fd_path[dir_len] = '/';
        memcpy(fd_path + dir_len + 1U, de->d_name, name_len + 1U);
        ssize_t n = readlink(fd_path, link, sizeof(link) - 1U);
        if (n <= 0) {
            continue;
        }
        link[n] = '\0';
        if (strncmp(link, "/dev/nvidia", 11) == 0) {
            found = 1;
            break;
        }
    }
    closedir(dir);
    return found;
}

static int process_is_monitor(pid_t pid)
{
    char path[PATH_MAX];
    char comm[128];
    snprintf(path, sizeof(path), "/proc/%ld/comm", (long)pid);
    if (read_first_line(path, comm, sizeof(comm)) != 0) {
        return 0;
    }
    return strcmp(comm, "nvidia-smi") == 0;
}

static unsigned int scan_nvidia_processes(nvmlProcessInfo_t *infos, unsigned int capacity)
{
    DIR *dir = opendir("/proc");
    if (dir == NULL) {
        return 0;
    }
    unsigned int found = 0;
    pid_t self = getpid();
    struct dirent *de = NULL;
    while ((de = readdir(dir)) != NULL) {
        if (!isdigit((unsigned char)de->d_name[0])) {
            continue;
        }
        char *end = NULL;
        long parsed = strtol(de->d_name, &end, 10);
        if (end == de->d_name || *end != '\0' || parsed <= 0 || parsed == (long)self) {
            continue;
        }
        pid_t pid = (pid_t)parsed;
        if (process_is_monitor(pid)) {
            continue;
        }
        if (!process_has_nvidia_fd(pid)) {
            continue;
        }
        if (infos != NULL && found < capacity) {
            memset(&infos[found], 0, sizeof(infos[found]));
            infos[found].pid = (unsigned int)pid;
            infos[found].usedGpuMemory = read_process_gpu_memory(pid);
        }
        found++;
        if (found >= MAX_NVML_PROCS && infos == NULL) {
            break;
        }
    }
    closedir(dir);
    return found;
}

static unsigned long long scan_total_process_gpu_memory(void)
{
    DIR *dir = opendir("/proc");
    if (dir == NULL) {
        return 0;
    }

    unsigned long long total = 0;
    pid_t self = getpid();
    struct dirent *de = NULL;
    while ((de = readdir(dir)) != NULL) {
        if (!isdigit((unsigned char)de->d_name[0])) {
            continue;
        }
        char *end = NULL;
        long parsed = strtol(de->d_name, &end, 10);
        if (end == de->d_name || *end != '\0' || parsed <= 0 || parsed == (long)self) {
            continue;
        }
        pid_t pid = (pid_t)parsed;
        if (process_is_monitor(pid) || !process_has_nvidia_fd(pid)) {
            continue;
        }
        unsigned long long bytes = read_process_gpu_memory(pid);
        if (bytes != NVML_VALUE_NOT_AVAILABLE) {
            total += bytes;
        }
    }
    closedir(dir);
    return total;
}

static nvmlReturn_t cuda_to_nvml(CUresult rc)
{
    switch (rc) {
    case CUDA_SUCCESS:
        return NVML_SUCCESS;
    case CUDA_ERROR_INVALID_VALUE:
    case CUDA_ERROR_INVALID_DEVICE:
    case CUDA_ERROR_INVALID_HANDLE:
        return NVML_ERROR_INVALID_ARGUMENT;
    case CUDA_ERROR_OUT_OF_MEMORY:
        return NVML_ERROR_MEMORY;
    case CUDA_ERROR_NOT_INITIALIZED:
        return NVML_ERROR_UNINITIALIZED;
    default:
        return NVML_ERROR_UNKNOWN;
    }
}

nvmlReturn_t nvmlInit(void)
{
    return nvmlInit_v2();
}

nvmlReturn_t nvmlInit_v2(void)
{
    CUresult rc = cuInit(0);
    if (rc != CUDA_SUCCESS) {
        g_initialized = 0;
        g_device_count = 0;
        return rc == CUDA_ERROR_NO_DEVICE ? NVML_ERROR_DRIVER_NOT_LOADED : cuda_to_nvml(rc);
    }
    int count = 0;
    rc = cuDeviceGetCount(&count);
    if (rc != CUDA_SUCCESS) {
        g_initialized = 0;
        g_device_count = 0;
        return cuda_to_nvml(rc);
    }
    if (count < 0) {
        count = 0;
    }
    if ((unsigned int)count > MAX_NVML_DEVICES) {
        count = (int)MAX_NVML_DEVICES;
    }
    g_device_count = (unsigned int)count;
    for (unsigned int i = 0; i < g_device_count; i++) {
        g_devices[i].index = i;
    }
    g_initialized = 1;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlShutdown(void)
{
    g_initialized = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount)
{
    return nvmlDeviceGetCount_v2(deviceCount);
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount)
{
    if (deviceCount == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (!g_initialized) {
        return NVML_ERROR_UNINITIALIZED;
    }
    *deviceCount = g_device_count;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device)
{
    return nvmlDeviceGetHandleByIndex_v2(index, device);
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device)
{
    if (device == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (!g_initialized) {
        return NVML_ERROR_UNINITIALIZED;
    }
    if (index >= g_device_count) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    *device = &g_devices[index];
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId, nvmlDevice_t *device)
{
    return nvmlDeviceGetHandleByPciBusId_v2(pciBusId, device);
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, nvmlDevice_t *device)
{
    if (pciBusId == NULL || device == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (!g_initialized) {
        return NVML_ERROR_UNINITIALIZED;
    }
    for (unsigned int i = 0; i < g_device_count; i++) {
        char bus_id[64] = {0};
        if (get_bus_id_for_index(i, bus_id, sizeof(bus_id)) == 0 &&
            strcasecmp(bus_id, pciBusId) == 0) {
            *device = &g_devices[i];
            return NVML_SUCCESS;
        }
    }
    return NVML_ERROR_NOT_FOUND;
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length)
{
    if (name == NULL || length == 0) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    CUdevice dev = 0;
    nvmlReturn_t rc = cuda_device_for_handle(device, &dev);
    if (rc != NVML_SUCCESS) {
        return rc;
    }
    return cuda_to_nvml(cuDeviceGetName(name, (int)length, dev));
}

nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci)
{
    if (pci == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    CUdevice dev = 0;
    nvmlReturn_t rc = cuda_device_for_handle(device, &dev);
    if (rc != NVML_SUCCESS) {
        return rc;
    }
    memset(pci, 0, sizeof(*pci));
    char bus_id[64] = {0};
    CUresult cu_rc = cuDeviceGetPCIBusId(bus_id, sizeof(bus_id), dev);
    if (cu_rc != CUDA_SUCCESS) {
        return cuda_to_nvml(cu_rc);
    }
    unsigned int domain = 0, bus = 0, devnum = 0, fn = 0;
    (void)parse_pci_bus_id(bus_id, &domain, &bus, &devnum, &fn);
    pci->domain = domain;
    pci->bus = bus;
    pci->device = devnum;
    snprintf(pci->busId, sizeof(pci->busId), "%s", bus_id);
    snprintf(pci->busIdLegacy, sizeof(pci->busIdLegacy), "%02x:%02x.%u", bus, devnum, fn);

    char path[PATH_MAX];
    unsigned int vendor = 0x10de;
    unsigned int pci_device = 0;
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/vendor", bus_id);
    (void)read_hex_file(path, &vendor);
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/device", bus_id);
    (void)read_hex_file(path, &pci_device);
    pci->pciDeviceId = (pci_device << 16) | vendor;

    unsigned int subsystem_vendor = 0;
    unsigned int subsystem_device = 0;
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/subsystem_vendor", bus_id);
    (void)read_hex_file(path, &subsystem_vendor);
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/subsystem_device", bus_id);
    (void)read_hex_file(path, &subsystem_device);
    pci->pciSubSystemId = (subsystem_device << 16) | subsystem_vendor;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory)
{
    if (memory == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    CUdevice dev = 0;
    nvmlReturn_t rc = cuda_device_for_handle(device, &dev);
    if (rc != NVML_SUCCESS) {
        return rc;
    }
    size_t total = 0;
    size_t free_bytes = 0;
    CUresult cu_rc = cuDeviceTotalMem(&total, dev);
    if (cu_rc != CUDA_SUCCESS) {
        return cuda_to_nvml(cu_rc);
    }
    unsigned long long process_used = scan_total_process_gpu_memory();
    if (process_used != 0) {
        free_bytes = process_used < total ? total - process_used : 0;
    } else {
        cu_rc = cuMemGetInfo(&free_bytes, &total);
        if (cu_rc != CUDA_SUCCESS) {
            free_bytes = total;
        }
    }
    memory->total = (unsigned long long)total;
    memory->free = (unsigned long long)free_bytes;
    memory->used = free_bytes < total ? (unsigned long long)(total - free_bytes) : 0ULL;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz)
{
    if (clockMHz == NULL || clockType >= NVML_CLOCK_COUNT) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    CUdevice dev = 0;
    nvmlReturn_t rc = cuda_device_for_handle(device, &dev);
    if (rc != NVML_SUCCESS) {
        return rc;
    }
    CUdevice_attribute attr = clockType == NVML_CLOCK_MEM ?
        CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE : CU_DEVICE_ATTRIBUTE_CLOCK_RATE;
    int khz = 0;
    CUresult cu_rc = cuDeviceGetAttribute(&khz, attr, dev);
    if (cu_rc != CUDA_SUCCESS || khz <= 0) {
        return NVML_ERROR_NOT_SUPPORTED;
    }
    *clockMHz = (unsigned int)(khz / 1000);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t *current, nvmlEnableState_t *pending)
{
    nvmlReturn_t rc = cuda_device_for_handle(device, NULL);
    if (rc != NVML_SUCCESS) {
        return rc;
    }
    if (current == NULL || pending == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    *current = NVML_FEATURE_DISABLED;
    *pending = NVML_FEATURE_DISABLED;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length)
{
    if (uuid == NULL || length == 0) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    CUdevice dev = 0;
    nvmlReturn_t rc = cuda_device_for_handle(device, &dev);
    if (rc != NVML_SUCCESS) {
        return rc;
    }
    char bus_id[64] = {0};
    if (cuDeviceGetPCIBusId(bus_id, sizeof(bus_id), dev) == CUDA_SUCCESS &&
        read_proc_gpu_field(bus_id, "GPU UUID", uuid, length) == 0) {
        return NVML_SUCCESS;
    }
    CUuuid cu_uuid;
    CUresult cu_rc = cuDeviceGetUuid(&cu_uuid, dev);
    if (cu_rc != CUDA_SUCCESS) {
        return cuda_to_nvml(cu_rc);
    }
    format_uuid_from_cuda(&cu_uuid, uuid, length);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed)
{
    nvmlReturn_t rc = cuda_device_for_handle(device, NULL);
    if (rc != NVML_SUCCESS) {
        return rc;
    }
    if (speed == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, unsigned int sensorType, unsigned int *temp)
{
    nvmlReturn_t rc = cuda_device_for_handle(device, NULL);
    if (rc != NVML_SUCCESS) {
        return rc;
    }
    if (temp == NULL || sensorType != NVML_TEMPERATURE_GPU) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t *utilization)
{
    nvmlReturn_t rc = cuda_device_for_handle(device, NULL);
    if (rc != NVML_SUCCESS) {
        return rc;
    }
    if (utilization == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    CUdevice dev = 0;
    rc = cuda_device_for_handle(device, &dev);
    if (rc != NVML_SUCCESS) {
        return rc;
    }

    size_t total = 0;
    if (cuDeviceTotalMem(&total, dev) != CUDA_SUCCESS || total == 0) {
        total = 0;
    }
    unsigned long long used = scan_total_process_gpu_memory();
    memset(utilization, 0, sizeof(*utilization));
    utilization->gpu = scan_total_process_gpu_util();
    utilization->memory = total != 0 ? clamp_percent((double)used * 100.0 / (double)total) : 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t *utilization,
                                             unsigned int *processSamplesCount,
                                             unsigned long long lastSeenTimeStamp)
{
    nvmlReturn_t rc = cuda_device_for_handle(device, NULL);
    if (rc != NVML_SUCCESS) {
        return rc;
    }
    if (processSamplesCount == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    unsigned int capacity = utilization != NULL ? *processSamplesCount : 0U;
    unsigned int found = 0;
    unsigned long long timestamp = now_timestamp_us();
    if (timestamp <= lastSeenTimeStamp) {
        timestamp = lastSeenTimeStamp + 1U;
    }

    DIR *dir = opendir("/proc");
    if (dir == NULL) {
        *processSamplesCount = 0;
        return NVML_SUCCESS;
    }

    pid_t self = getpid();
    struct dirent *de = NULL;
    while ((de = readdir(dir)) != NULL) {
        if (!isdigit((unsigned char)de->d_name[0])) {
            continue;
        }
        char *end = NULL;
        long parsed = strtol(de->d_name, &end, 10);
        if (end == de->d_name || *end != '\0' || parsed <= 0 || parsed == (long)self) {
            continue;
        }
        pid_t pid = (pid_t)parsed;
        if (process_is_monitor(pid) || !process_has_nvidia_fd(pid)) {
            continue;
        }
        if (utilization != NULL && found < capacity) {
            unsigned long long bytes = read_process_gpu_memory(pid);
            memset(&utilization[found], 0, sizeof(utilization[found]));
            utilization[found].pid = (unsigned int)pid;
            utilization[found].timeStamp = timestamp;
            utilization[found].smUtil = estimate_process_gpu_util(pid);
            utilization[found].memUtil = bytes != NVML_VALUE_NOT_AVAILABLE && bytes != 0 ? 1U : 0U;
        }
        found++;
        if (found >= MAX_NVML_PROCS && utilization == NULL) {
            break;
        }
    }
    closedir(dir);

    *processSamplesCount = found;
    if (utilization == NULL || capacity < found) {
        return found == 0 ? NVML_SUCCESS : NVML_ERROR_INSUFFICIENT_SIZE;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power)
{
    nvmlReturn_t rc = cuda_device_for_handle(device, NULL);
    if (rc != NVML_SUCCESS) {
        return rc;
    }
    if (power == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos)
{
    return nvmlDeviceGetComputeRunningProcesses_v2(device, infoCount, infos);
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos)
{
    nvmlReturn_t rc = cuda_device_for_handle(device, NULL);
    if (rc != NVML_SUCCESS) {
        return rc;
    }
    if (infoCount == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    unsigned int capacity = infos != NULL ? *infoCount : 0U;
    unsigned int found = scan_nvidia_processes(infos, capacity);
    if (infos != NULL && capacity < found) {
        *infoCount = found;
        return NVML_ERROR_INSUFFICIENT_SIZE;
    }
    *infoCount = found;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos)
{
    return nvmlDeviceGetComputeRunningProcesses_v2(device, infoCount, infos);
}

nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link,
                                                 nvmlIntNvLinkDeviceType_t *pNvLinkDeviceType)
{
    (void)link;
    nvmlReturn_t rc = cuda_device_for_handle(device, NULL);
    if (rc != NVML_SUCCESS) {
        return rc;
    }
    if (pNvLinkDeviceType == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    *pNvLinkDeviceType = NVML_NVLINK_DEVICE_TYPE_UNKNOWN;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci)
{
    (void)link;
    nvmlReturn_t rc = cuda_device_for_handle(device, NULL);
    if (rc != NVML_SUCCESS) {
        return rc;
    }
    if (pci == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    memset(pci, 0, sizeof(*pci));
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length)
{
    if (version == NULL || length == 0) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (extract_driver_version(version, length) != 0) {
        copy_string(version, length, "N/A");
        return NVML_ERROR_NOT_SUPPORTED;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion(int *cudaDriverVersion)
{
    return nvmlSystemGetCudaDriverVersion_v2(cudaDriverVersion);
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *cudaDriverVersion)
{
    if (cudaDriverVersion == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    CUresult rc = cuDriverGetVersion(cudaDriverVersion);
    return cuda_to_nvml(rc);
}

nvmlReturn_t nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, nvmlGpuFabricInfoV_t *gpuFabricInfo)
{
    nvmlReturn_t rc = cuda_device_for_handle(device, NULL);
    if (rc != NVML_SUCCESS) {
        return rc;
    }
    if (gpuFabricInfo == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    memset(gpuFabricInfo, 0, sizeof(*gpuFabricInfo));
    gpuFabricInfo->state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;
    gpuFabricInfo->status = NVML_ERROR_NOT_SUPPORTED;
    return NVML_ERROR_NOT_SUPPORTED;
}

const char *nvmlErrorString(nvmlReturn_t result)
{
    switch (result) {
    case NVML_SUCCESS: return "Success";
    case NVML_ERROR_UNINITIALIZED: return "Uninitialized";
    case NVML_ERROR_INVALID_ARGUMENT: return "Invalid Argument";
    case NVML_ERROR_NOT_SUPPORTED: return "Not Supported";
    case NVML_ERROR_NO_PERMISSION: return "No Permission";
    case NVML_ERROR_ALREADY_INITIALIZED: return "Already Initialized";
    case NVML_ERROR_NOT_FOUND: return "Not Found";
    case NVML_ERROR_INSUFFICIENT_SIZE: return "Insufficient Size";
    case NVML_ERROR_INSUFFICIENT_POWER: return "Insufficient Power";
    case NVML_ERROR_DRIVER_NOT_LOADED: return "Driver Not Loaded";
    case NVML_ERROR_TIMEOUT: return "Timeout";
    case NVML_ERROR_IRQ_ISSUE: return "IRQ Issue";
    case NVML_ERROR_LIBRARY_NOT_FOUND: return "Library Not Found";
    case NVML_ERROR_FUNCTION_NOT_FOUND: return "Function Not Found";
    case NVML_ERROR_CORRUPTED_INFOROM: return "Corrupted InfoROM";
    case NVML_ERROR_GPU_IS_LOST: return "GPU Is Lost";
    case NVML_ERROR_RESET_REQUIRED: return "Reset Required";
    case NVML_ERROR_OPERATING_SYSTEM: return "Operating System Error";
    case NVML_ERROR_LIB_RM_VERSION_MISMATCH: return "RM Version Mismatch";
    case NVML_ERROR_IN_USE: return "In Use";
    case NVML_ERROR_MEMORY: return "Memory Error";
    case NVML_ERROR_NO_DATA: return "No Data";
    case NVML_ERROR_VGPU_ECC_NOT_SUPPORTED: return "vGPU ECC Not Supported";
    default: return "Unknown Error";
    }
}
