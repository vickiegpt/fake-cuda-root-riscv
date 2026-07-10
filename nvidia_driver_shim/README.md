# Lanxin RISC-V NVIDIA libcuda shim

This is the native RISC-V `libcuda.so.1` layer for Lanxin bring-up. It opens the real NVIDIA kernel driver nodes (`/dev/nvidiactl`, `/dev/nvidia0`, and UVM nodes when present), reports the RTX 5090 D found under `/proc/driver/nvidia/gpus`, and implements enough CUDA Driver API plus RM ioctl scaffolding for device/context/memory/copy/channel sampling.

Current boundary:

- Real: NVIDIA device-node open, RM client/device/subdevice alloc, RM-backed sysmem alloc/map/free for `cuMemAlloc`, device enumeration, context handles, stream/event handles, CUDA memory accounting, `cuMemcpy*`, `cuMemset*`, pointer attributes, module/link/library/kernel code-load handles, and `cuGetProcAddress`.
- Real management surface: a minimal `libnvidia-ml.so.1` plus `bin/nvidia-smi` now report driver/CUDA version, GPU name, UUID, PCI bus ID, memory totals/free/used from the shim accounting path, PCIe link information from sysfs, and process discovery by scanning `/proc/*/fd` for NVIDIA device nodes.
- Real RM channel scaffold: `NV01_MEMORY_VIRTUAL` GPU VA, notifier sysmem + error ctxdma, GPFIFO sysmem CPU/GPU mapping, client-allocated UserD sysmem, `BLACKWELL_CHANNEL_GPFIFO_B` alloc, bind, schedule, work-submit-token, and a GPFIFO NOP submit by writing UserD `GPPut`.
- Real RM object/pushbuffer scaffold: compute object allocation probes `BLACKWELL_COMPUTE_B/A`, falls back through Hopper/Ampere classes, calls `NV906F_CTRL_GET_CLASS_ENGINEID`, writes a C46F-format compute `SET_OBJECT` + `NO_OPERATION` + `PIPE_NOP` pushbuffer, and submits it through the existing GPFIFO ring.
- Real staging for the next launch layer: `cuLaunchKernel` now stages a launch packet into RM system memory mapped into the channel VASpace: module code-object bytes, a QMD-shaped descriptor, captured `CU_LAUNCH_PARAM_BUFFER_*` argument bytes, and a completion record. The GPFIFO `GPPut` update is the current doorbell.
- Provisional: `cuModuleLoad*`, `cuLink*`, `cuLibrary*`, and `cuKernel*` accept PTX/cubin/fatbin-like payloads and route functions/kernels into the launch scaffold, but the shim does not yet execute CUDA code objects/SASS with a verified NVIDIA hardware QMD. By default, `cuLaunchKernel` stages the launch packet, submits the safe compute `SET_OBJECT` pushbuffer through RM, and returns `CUDA_SUCCESS` so loader/runtime smoke tests can run end-to-end. `LANXIN_NVIDIA_CUDA_STRICT_LAUNCH=1` keeps the same RM submit but returns `CUDA_ERROR_NOT_SUPPORTED` to preserve the honest bring-up boundary.
- Known boundary: simple CPU BAR1 mapping of `NV01_MEMORY_LOCAL_USER` VRAM returns `NV_ERR_NOT_SUPPORTED` on this driver path, so current `cuMemAlloc` uses RM-backed sysmem rather than mappable VRAM.

Build:

```sh
cd /home/ubuntu/fake_cuda
./nvidia_driver_shim/build.sh
LD_LIBRARY_PATH=/home/ubuntu/fake_cuda/lib64 ./nvidia_driver_shim/build/cuda_probe
./bin/nvidia-smi
./bin/nvidia-smi -L
./bin/nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,memory.free --format=csv,noheader,nounits
LANXIN_NVIDIA_CUDA_TRACE=1 ./nvidia_driver_shim/build/launch_probe
LANXIN_NVIDIA_CUDA_TRACE=1 LANXIN_NVIDIA_CUDA_QMD_SUBMIT=1 ./nvidia_driver_shim/build/launch_probe
LANXIN_NVIDIA_CUDA_TRACE=1 LANXIN_NVIDIA_CUDA_STRICT_LAUNCH=1 ./nvidia_driver_shim/build/launch_probe
LANXIN_NVIDIA_CUDA_TRACE=1 ./nvidia_driver_shim/build/api_probe
./nvidia_driver_shim/build/channel_probe
```

Useful environment overrides:

- `LANXIN_NVIDIA_CUDA_TRACE=1` prints shim calls.
- `LANXIN_NVIDIA_CUDA_TOTAL_MEM_MB=32768` overrides reported memory size.
- `LANXIN_NVIDIA_CUDA_SM_COUNT=170` overrides reported SM count.
- `LANXIN_NVIDIA_CUDA_NOOP_KERNEL=1` makes `cuLaunchKernel` submit a real RM GPFIFO NOP and return success.
- `LANXIN_NVIDIA_CUDA_RM_SUBMIT=1` makes `cuLaunchKernel` submit the same RM GPFIFO NOP but keep returning `CUDA_ERROR_NOT_SUPPORTED`.
- `LANXIN_NVIDIA_CUDA_PB_SUBMIT=0` disables the default compute pushbuffer submit path.
- `LANXIN_NVIDIA_CUDA_QMD_STAGE=0` disables QMD/code/params/completion staging. Staging is enabled by default.
- `LANXIN_NVIDIA_CUDA_QMD_SUBMIT=1` requests the experimental QMD pushbuffer path. Without verified `LANXIN_NVIDIA_CUDA_QMD_METHOD_LO` and `LANXIN_NVIDIA_CUDA_QMD_METHOD_HI` method offsets, the shim stages QMD state but still submits only the safe `SET_OBJECT` pushbuffer.
- `LANXIN_NVIDIA_CUDA_WAIT_COMPLETION=1` polls the staged completion record after doorbell. This is expected to time out until a real hardware QMD launch writes completion.
- `LANXIN_NVIDIA_CUDA_STRICT_LAUNCH=1` makes `cuLaunchKernel` try the RM submit path but report `CUDA_ERROR_NOT_SUPPORTED` until real QMD/SASS launch is implemented.
- `LANXIN_NVIDIA_CUDA_PB_SUBMIT=1` explicitly selects the compute `SET_OBJECT` pushbuffer path. This is also the default unless disabled.
- `LANXIN_NVIDIA_CUDA_RM=0` disables the RM-backed allocation/channel path and falls back to host-backed shim memory where possible.
