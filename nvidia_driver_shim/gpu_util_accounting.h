#ifndef LANXIN_GPU_UTIL_ACCOUNTING_H
#define LANXIN_GPU_UTIL_ACCOUNTING_H

#include <stdint.h>

#define LANXIN_GPU_UTIL_ACCOUNTING_MAGIC 0x4c58475554494c31ULL
#define LANXIN_GPU_UTIL_ACCOUNTING_VERSION 1U
#define LANXIN_GPU_UTIL_INTERVAL_CAPACITY 1024U

struct lanxin_gpu_busy_interval {
    uint64_t start_ns;
    uint64_t end_ns;
};

struct lanxin_gpu_util_accounting {
    uint64_t magic;
    uint32_t version;
    uint32_t size;
    uint64_t sequence;
    uint64_t pid;
    uint64_t epoch_ns;
    uint64_t active_start_ns;
    uint64_t last_submit_ns;
    uint64_t last_complete_ns;
    uint64_t submits;
    uint64_t completes;
    uint64_t timeouts;
    uint32_t inflight;
    uint32_t interval_head;
    uint32_t interval_count;
    uint32_t reserved;
    struct lanxin_gpu_busy_interval intervals[LANXIN_GPU_UTIL_INTERVAL_CAPACITY];
};

#endif
