# Lanxin RISC-V NVIDIA libcuda shim

This is the native RISC-V `libcuda.so.1` layer for Lanxin bring-up. It opens the real NVIDIA kernel driver nodes (`/dev/nvidiactl`, `/dev/nvidia0`, and UVM nodes when present), reports the RTX 5090 D found under `/proc/driver/nvidia/gpus`, and implements enough CUDA Driver API plus RM ioctl scaffolding for device/context/memory/copy/channel sampling.

Current boundary:

- Real: NVIDIA device-node open, RM client/device/subdevice alloc, RM-backed sysmem alloc/map/free for `cuMemAlloc`, device enumeration, context handles, stream/event handles, CUDA memory accounting, `cuMemcpy*`, `cuMemset*`, pointer attributes, and `cuGetProcAddress`.
- Real RM channel scaffold: `NV01_MEMORY_VIRTUAL` GPU VA, notifier sysmem + error ctxdma, GPFIFO sysmem CPU/GPU mapping, client-allocated UserD sysmem, `BLACKWELL_CHANNEL_GPFIFO_B` alloc, bind, schedule, work-submit-token, and a GPFIFO NOP submit by writing UserD `GPPut`.
- Real RM object/pushbuffer scaffold: compute object allocation probes `BLACKWELL_COMPUTE_B/A`, falls back through Hopper/Ampere classes, calls `NV906F_CTRL_GET_CLASS_ENGINEID`, writes a C46F-format compute `SET_OBJECT` + `NO_OPERATION` + `PIPE_NOP` pushbuffer, and submits it through the existing GPFIFO ring.
- Provisional: `cuLaunchKernel` does not execute CUDA code objects/SASS/QMD yet. With `LANXIN_NVIDIA_CUDA_NOOP_KERNEL=1`, it performs a real RM submission and returns success. With `LANXIN_NVIDIA_CUDA_RM_SUBMIT=1` or `LANXIN_NVIDIA_CUDA_PB_SUBMIT=1`, it submits through RM but still returns `CUDA_ERROR_NOT_SUPPORTED`.
- Known boundary: simple CPU BAR1 mapping of `NV01_MEMORY_LOCAL_USER` VRAM returns `NV_ERR_NOT_SUPPORTED` on this driver path, so current `cuMemAlloc` uses RM-backed sysmem rather than mappable VRAM.

Build:

```sh
cd /home/ubuntu/fake_cuda
./nvidia_driver_shim/build.sh
LD_LIBRARY_PATH=/home/ubuntu/fake_cuda/lib64 ./nvidia_driver_shim/build/cuda_probe
LANXIN_NVIDIA_CUDA_TRACE=1 LANXIN_NVIDIA_CUDA_NOOP_KERNEL=1 ./nvidia_driver_shim/build/launch_probe
LANXIN_NVIDIA_CUDA_TRACE=1 LANXIN_NVIDIA_CUDA_PB_SUBMIT=1 ./nvidia_driver_shim/build/launch_probe
./nvidia_driver_shim/build/channel_probe
```

Useful environment overrides:

- `LANXIN_NVIDIA_CUDA_TRACE=1` prints shim calls.
- `LANXIN_NVIDIA_CUDA_TOTAL_MEM_MB=32768` overrides reported memory size.
- `LANXIN_NVIDIA_CUDA_SM_COUNT=170` overrides reported SM count.
- `LANXIN_NVIDIA_CUDA_NOOP_KERNEL=1` makes `cuLaunchKernel` submit a real RM GPFIFO NOP and return success.
- `LANXIN_NVIDIA_CUDA_RM_SUBMIT=1` makes `cuLaunchKernel` submit the same RM GPFIFO NOP but keep returning `CUDA_ERROR_NOT_SUPPORTED`.
- `LANXIN_NVIDIA_CUDA_PB_SUBMIT=1` makes `cuLaunchKernel` allocate/query a compute object, submit a compute `SET_OBJECT` pushbuffer, and keep returning `CUDA_ERROR_NOT_SUPPORTED` until real QMD/SASS launch is implemented.
- `LANXIN_NVIDIA_CUDA_RM=0` disables the RM-backed allocation/channel path and falls back to host-backed shim memory where possible.
