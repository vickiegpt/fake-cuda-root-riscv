#!/bin/sh
#
# Profile the Lanxin NVIDIA kernel path around a supplied workload.
# Run as root so tracefs and PCI configuration are readable.

set -eu

usage()
{
    echo "usage: sudo $0 OUTPUT_DIR -- COMMAND [ARG ...]" >&2
    exit 2
}

[ "$#" -ge 3 ] || usage
out=$1
shift
[ "$1" = "--" ] || usage
shift

tracefs=/sys/kernel/tracing
[ -d "$tracefs" ] || tracefs=/sys/kernel/debug/tracing
[ -d "$tracefs" ] || {
    echo "tracefs is not mounted" >&2
    exit 1
}

if [ "$(id -u)" -ne 0 ]; then
    echo "run this script with sudo" >&2
    exit 1
fi

mkdir -p "$out"
out=$(cd "$out" && pwd)

cleanup()
{
    trace-cmd reset >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

snapshot()
{
    label=$1
    cp /proc/interrupts "$out/interrupts.$label"
    cp /proc/softirqs "$out/softirqs.$label"
    cp /proc/stat "$out/proc_stat.$label"
    cp /proc/vmstat "$out/vmstat.$label"
    lspci -vv -s 0001:01:00.0 > "$out/lspci.$label" 2>&1 || true
}

{
    date -Ins
    uname -a
    echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
    echo "command=$*"
    echo "tracefs=$tracefs"
    for dev in /sys/bus/pci/devices/*; do
        [ "$(cat "$dev/vendor" 2>/dev/null || true)" = 0x10de ] || continue
        echo "pci=${dev##*/} class=$(cat "$dev/class") irq=$(cat "$dev/irq")"
    done
} > "$out/metadata.txt"

snapshot before

start_ns=$(date +%s%N)
set +e
/usr/bin/time -v -o "$out/workload.time" \
    "$@" > "$out/workload.stdout" 2> "$out/workload.stderr"
status=$?
set -e
end_ns=$(date +%s%N)

snapshot after
echo "$status" > "$out/workload.status"
awk -v start="$start_ns" -v end="$end_ns" \
    'BEGIN { printf "%.6f\n", (end - start) / 1000000000 }' \
    > "$out/workload.wall_seconds"

perf_bin=
for candidate in /usr/lib/linux-tools/*/perf /usr/bin/perf; do
    if [ -x "$candidate" ] && "$candidate" version >/dev/null 2>&1; then
        perf_bin=$candidate
    fi
done

if [ -n "$perf_bin" ]; then
    set +e
    "$perf_bin" record -e cpu-clock -F 999 -g \
        -o "$out/perf.data" -- "$@" \
        > "$out/perf.stdout" 2> "$out/perf.stderr"
    perf_status=$?
    set -e
    echo "$perf_status" > "$out/perf.status"
    if [ "$perf_status" -eq 0 ]; then
        "$perf_bin" report --stdio --call-graph none --no-children \
            --sort dso,symbol --percent-limit 0.05 \
            -i "$out/perf.data" > "$out/perf.flat.txt" 2>&1 || true
        "$perf_bin" report --stdio --call-graph none --children \
            --sort dso,symbol --percent-limit 0.5 \
            -i "$out/perf.data" > "$out/perf.children.txt" 2>&1 || true
    fi
fi

trace_args=
for event in \
    irq:irq_handler_entry irq:irq_handler_exit \
    irq:softirq_entry irq:softirq_exit \
    iommu:map iommu:unmap iommu:io_page_fault \
    dma_fence:dma_fence_wait_start dma_fence:dma_fence_wait_end \
    dma_fence:dma_fence_signaled
do
    group=${event%%:*}
    name=${event#*:}
    if [ -d "$tracefs/events/$group/$name" ]; then
        trace_args="$trace_args -e $event"
    fi
done

set +e
# trace_args is intentionally word-split into trace-cmd's repeated -e options.
# shellcheck disable=SC2086
trace-cmd record -o "$out/trace.dat" $trace_args -- "$@" \
    > "$out/trace.stdout" 2> "$out/trace.stderr"
trace_status=$?
set -e
echo "$trace_status" > "$out/trace.status"
if [ "$trace_status" -eq 0 ]; then
    trace-cmd report -i "$out/trace.dat" > "$out/trace.txt" 2>&1 || true
fi

python3 - "$out" <<'PY'
import collections
import pathlib
import re
import statistics
import sys

out = pathlib.Path(sys.argv[1])

def rows(path):
    result = {}
    for line in path.read_text().splitlines():
        match = re.match(r"\s*(\S+):\s+(.*)", line)
        if not match:
            continue
        name, values = match.groups()
        nums = [int(value) for value in values.split() if value.isdigit()]
        result[name] = nums
    return result

def write_delta(kind):
    before = rows(out / f"{kind}.before")
    after = rows(out / f"{kind}.after")
    with (out / f"{kind}.delta.tsv").open("w") as stream:
        stream.write("name\ttotal_delta\tper_cpu_delta\n")
        for name in sorted(set(before) | set(after)):
            lhs = before.get(name, [])
            rhs = after.get(name, [])
            width = max(len(lhs), len(rhs))
            lhs += [0] * (width - len(lhs))
            rhs += [0] * (width - len(rhs))
            delta = [r - l for l, r in zip(lhs, rhs)]
            if any(delta):
                stream.write(
                    f"{name}\t{sum(delta)}\t{','.join(map(str, delta))}\n"
                )

write_delta("interrupts")
write_delta("softirqs")

trace_path = out / "trace.txt"
if trace_path.exists():
    active = collections.defaultdict(list)
    durations = collections.defaultdict(list)
    other_events = collections.Counter()
    pattern = re.compile(
        r"\[(\d+)\]\s+([0-9.]+): irq_handler_(entry|exit):\s+"
        r"irq=(\d+)(?: name=(.*))?"
    )
    for line in trace_path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            cpu, timestamp, kind, irq, name = match.groups()
            key = (cpu, irq)
            timestamp = float(timestamp)
            if kind == "entry":
                active[key].append((timestamp, (name or "unknown").strip()))
            elif active[key]:
                started, irq_name = active[key].pop()
                durations[irq_name].append((timestamp - started) * 1.0e6)
        if " iommu_map:" in line:
            other_events["iommu_map"] += 1
        elif " iommu_unmap:" in line:
            other_events["iommu_unmap"] += 1
        elif " io_page_fault:" in line:
            other_events["iommu_page_fault"] += 1
        elif " dma_fence_" in line:
            other_events["dma_fence"] += 1

    with (out / "irq_latency.tsv").open("w") as stream:
        stream.write("name\tcount\ttotal_us\tavg_us\tp99_us\tmax_us\n")
        for name, values in sorted(
            durations.items(), key=lambda item: sum(item[1]), reverse=True
        ):
            ordered = sorted(values)
            p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]
            stream.write(
                f"{name}\t{len(values)}\t{sum(values):.3f}\t"
                f"{statistics.mean(values):.3f}\t{p99:.3f}\t"
                f"{max(values):.3f}\n"
            )

    with (out / "kernel_event_counts.tsv").open("w") as stream:
        stream.write("event\tcount\n")
        for name in ("iommu_map", "iommu_unmap", "iommu_page_fault", "dma_fence"):
            stream.write(f"{name}\t{other_events[name]}\n")
PY

exit "$status"
