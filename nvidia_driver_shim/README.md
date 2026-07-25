# Lanxin RISC-V NVIDIA libcuda shim

This is the native RISC-V `libcuda.so.1` layer for Lanxin bring-up. It opens the real NVIDIA kernel driver nodes (`/dev/nvidiactl`, `/dev/nvidia0`, and UVM nodes when present), reports the RTX 5090 D found under `/proc/driver/nvidia/gpus`, and implements enough CUDA Driver API plus RM ioctl scaffolding for device/context/memory/copy/channel sampling.

Current boundary:

- Real: NVIDIA device-node open, RM client/device/subdevice alloc, RM-backed sysmem alloc/map/free for `cuMemAlloc`, device enumeration, context handles, stream/event handles, CUDA memory accounting, `cuMemcpy*`, `cuMemset*`, pointer attributes, module/link/library/kernel code-load handles, and `cuGetProcAddress`.
- Real management surface: a minimal `libnvidia-ml.so.1` plus `bin/nvidia-smi` now report driver/CUDA version, GPU name, UUID, PCI bus ID, memory totals/free/used from the shim accounting path, PCIe link information from sysfs, and process discovery by scanning `/proc/*/fd` for NVIDIA device nodes.
- Real RM channel path: `NV01_MEMORY_VIRTUAL` GPU VA, notifier sysmem + error ctxdma, write-combined GPFIFO CPU/GPU mapping, persistent local-memory UserD, `BLACKWELL_USERMODE_A` doorbell mapping, `BLACKWELL_CHANNEL_GPFIFO_B` allocation/bind/schedule, refreshed work-submit tokens, and DoorbellKickoff-style submission by writing UserD `GPPut` then the USERMODE doorbell token. UserD and USERMODE use independent `/dev/nvidia0` file descriptions, which is required by this RM path.
- Real RM object/pushbuffer scaffold: compute object allocation probes `BLACKWELL_COMPUTE_B/A`, falls back through Hopper/Ampere classes, calls `NV906F_CTRL_GET_CLASS_ENGINEID`, writes a C46F-format compute `SET_OBJECT` + `NO_OPERATION` + `PIPE_NOP` pushbuffer, submits a paired progress-tracker semaphore pushbuffer, and verifies HOST consumption through a completion record.
- Real Blackwell launch path: `cuLaunchKernel` stages code, arguments, a 384-byte QMD 5.0 descriptor in a 64-slot ring, and completion state into RM memory mapped into the channel VASpace. It parses ELF `.text.<kernel>`, `.nv.info.<kernel>`, `.nv.constant0.<kernel>`, `EIATTR_PARAM_CBANK`, and `EIATTR_KPARAM_INFO`, so both standard `kernelParams` and `CU_LAUNCH_PARAM_BUFFER_*` ABIs use the cubin's declared argument layout. QMD submission uses Blackwell `SEND_PCAS_A` plus `SEND_SIGNALING_PCAS2_B`; a HOST WFI semaphore verifies that all preceding compute work completed.
- Verified hardware result: an official CUDA 12.9 `ptxas` SM120 cubin launched through this path writes `42` to an RM-allocated GPU-visible pointer with both argument ABIs, including repeated launches. The transitional Blackwell loader patches the private global descriptor bit in staged simple global-memory SASS; this is enough for the probe but is not a substitute for complete CUDA relocation and descriptor handling.
- Remaining performance boundary: `libcublas_nvidia.so.12` is still a CPU implementation and the shim does not yet implement the full CUDA stream, event, graph, texture, relocation, tensor-core, or cuBLAS semantics needed for RTX 5090 GEMM/LLM peak throughput. The verified path establishes real SM execution and completion, not theoretical tensor throughput.
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
LANXIN_NVIDIA_CUDA_TRACE=1 LANXIN_NVIDIA_CUDA_WAIT_COMPLETION=1 ./nvidia_driver_shim/build/launch_probe
LANXIN_NVIDIA_CUDA_TRACE=1 LANXIN_NVIDIA_CUDA_QMD_SUBMIT=1 LANXIN_NVIDIA_CUDA_STRICT_LAUNCH=1 ./nvidia_driver_shim/build/launch_probe
./bin/ptxas -arch=sm_120 -o /tmp/sm120_store_probe.cubin nvidia_driver_shim/sm120_store_probe.ptx
LD_LIBRARY_PATH=/home/ubuntu/fake_cuda/lib64 LANXIN_NVIDIA_CUDA_QMD_SUBMIT=1 LANXIN_NVIDIA_CUDA_STRICT_LAUNCH=1 ./nvidia_driver_shim/build/sm120_qmd_probe /tmp/sm120_store_probe.cubin 10000
LD_LIBRARY_PATH=/home/ubuntu/fake_cuda/lib64 LANXIN_NVIDIA_CUDA_QMD_SUBMIT=1 LANXIN_NVIDIA_CUDA_STRICT_LAUNCH=1 ./nvidia_driver_shim/build/sm120_qmd_probe /tmp/sm120_store_probe.cubin --extra 10000
LANXIN_NVIDIA_CUDA_TRACE=1 ./nvidia_driver_shim/build/api_probe
LANXIN_NVIDIA_CUDA_TRACE=1 ./nvidia_driver_shim/build/module_image_probe
LANXIN_NVIDIA_CUDA_TRACE=1 LANXIN_NVIDIA_CUDA_MODULE_IMAGE_LAUNCH=1 LANXIN_NVIDIA_CUDA_WAIT_COMPLETION=1 ./nvidia_driver_shim/build/module_image_probe
LANXIN_NVIDIA_CUDA_TRACE=1 LANXIN_NVIDIA_CUDA_MODULE_IMAGE_LAUNCH=1 LANXIN_NVIDIA_CUDA_MODULE_IMAGE_LAUNCH_SECOND=1 LANXIN_NVIDIA_CUDA_CODE_STAGE_TEXT=1 LANXIN_NVIDIA_CUDA_WAIT_COMPLETION=1 ./nvidia_driver_shim/build/module_image_probe
LANXIN_NVIDIA_CUDA_TRACE=1 ./nvidia_driver_shim/build/cubin_launch_probe /path/to/kernel.cubin kernel_name
LANXIN_NVIDIA_CUDA_TRACE=1 LANXIN_NVIDIA_CUDA_CODE_STAGE_TEXT=1 LANXIN_NVIDIA_CUDA_CUBIN_LAUNCH=1 ./nvidia_driver_shim/build/cubin_launch_probe /path/to/kernel.cubin kernel_name
./nvidia_driver_shim/llm_demo.sh
LANXIN_LLM_TRACE=1 ./nvidia_driver_shim/llm_demo.sh
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
- `LANXIN_NVIDIA_CUDA_QMD_SUBMIT=1` enables hardware QMD submission. Blackwell uses QMD 5.0 and PCAS2; older classes retain the QMD 1.6 path.
- `LANXIN_NVIDIA_CUDA_QMD_RELEASE=1` additionally enables and verifies the QMD release semaphore. Blackwell defaults to the faster HOST WFI completion path; older classes default to QMD release. Set it to `0` to force HOST WFI on any class.
- `LANXIN_NVIDIA_CUDA_QMD_REQUIRE_PCAS=0` clears QMDV01_06 bit 204 for A/B testing. The default sets `REQUIRE_SCHEDULING_PCAS` because this path submits through PCAS.
- `LANXIN_NVIDIA_CUDA_QMD_PROGRAM_OFFSET=0x...` overrides the parsed `.text.<kernel>` file offset written to QMD `PROGRAM_OFFSET`.
- `LANXIN_NVIDIA_CUDA_QMD_LOCAL_MEM_BYTES=0` overrides parsed `.nv.info.<kernel>` local-memory bytes.
- `LANXIN_NVIDIA_CUDA_CBANK_STACK_BYTES=0x...` independently overrides the SM120 CUDA constant-bank stack pointer for local-memory ABI diagnostics without changing QMD or SLM allocation size.
- `LANXIN_NVIDIA_CUDA_SLM_BYTES_PER_LANE=0x...` independently overrides the physical SLM backing stride without changing the QMD local-memory size or constant-bank stack pointer.
- `LANXIN_NVIDIA_CUDA_MAX_WARPS_PER_SM=48` overrides the Blackwell SLM sizing input. The default matches SM120's architectural maximum of 48 resident warps per SM.
- `LANXIN_NVIDIA_CUDA_QMD_REGISTER_COUNT=32` and `LANXIN_NVIDIA_CUDA_QMD_SASS_VERSION=120` override parsed cubin metadata.
- `LANXIN_NVIDIA_CUDA_CODE_STAGE_TEXT=1` stages only `.text.<kernel>` bytes for the selected function and writes QMD `PROGRAM_OFFSET=0` by default. Without it, the shim stages the full ELF/cubin image and uses the parsed file offset.
- `LANXIN_NVIDIA_CUDA_REQUIRE_CUBIN_METADATA=1` makes `cubin_launch_probe` fail when the named kernel did not parse any cubin metadata.
- `LANXIN_NVIDIA_CUDA_IMAGE_SCAN_MAX_BYTES=67108864` caps loader-side in-memory image probing for `cuModuleLoadData`.
- `LANXIN_NVIDIA_CUDA_CODE_STAGE_MAX_BYTES=16777216` caps how many module image bytes are copied into RM-mapped code-object staging memory.
- `LANXIN_NVIDIA_CUDA_WAIT_COMPLETION=1` polls the staged completion record after doorbell. Strict Blackwell QMD launches use a HOST WFI semaphore by default and therefore return only after the submitted kernel has completed.
- `LANXIN_NVIDIA_CUDA_DOORBELL=0` disables USERMODE doorbell allocation and falls back to UserD-only kickoff for debugging.
- `LANXIN_NVIDIA_CUDA_PROGRESS_WFI=1` makes the progress-tracker semaphore release use WFI.
- `LANXIN_NVIDIA_CUDA_STRICT_LAUNCH=1` requires the selected RM submission and completion path to succeed instead of masking a launch failure.
- `LANXIN_NVIDIA_CUDA_PB_SUBMIT=1` explicitly selects the compute `SET_OBJECT` pushbuffer path. This is also the default unless disabled.
- `LANXIN_NVIDIA_CUDA_RM=0` disables the RM-backed allocation/channel path and falls back to host-backed shim memory where possible.
