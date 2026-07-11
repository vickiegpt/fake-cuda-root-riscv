#!/usr/bin/env bash
set -euo pipefail

FAKE_CUDA_ROOT="${FAKE_CUDA_ROOT:-/home/ubuntu/fake_cuda}"
LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-/home/ubuntu/Documents/llama.cpp}"
LLAMA_BIN="${LLAMA_BIN:-$LLAMA_CPP_ROOT/build-lanxin-nvidia/bin/llama-cli}"
MODEL="${MODEL:-/home/ubuntu/Documents/pacc-llama.cpp/models/e2e/pacc-random-llama-f32.gguf}"
PROMPT="${PROMPT:-Lanxin fake CUDA LLM demo:}"
TOKENS="${TOKENS:-16}"
THREADS="${THREADS:-4}"
CTX_SIZE="${CTX_SIZE:-128}"
GPU_LAYERS="${GPU_LAYERS:-1}"
CPU_LOG="${CPU_LOG:-/tmp/lanxin_llm_demo_cpu.log}"
CUDA_LOG="${CUDA_LOG:-/tmp/lanxin_llm_demo_cuda.log}"

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "missing llama-cli: $LLAMA_BIN" >&2
  exit 2
fi
if [[ ! -f "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  exit 2
fi

export LD_LIBRARY_PATH="$FAKE_CUDA_ROOT/lib64:$LLAMA_CPP_ROOT/build-lanxin-nvidia/bin:${LD_LIBRARY_PATH:-}"

common_args=(
  -m "$MODEL"
  -p "$PROMPT"
  -n "$TOKENS"
  -c "$CTX_SIZE"
  -t "$THREADS"
  -st
  --simple-io
  --no-display-prompt
  --no-warmup
  --no-perf
)

echo "== fake CUDA device list =="
"$LLAMA_BIN" --list-devices

echo
echo "== CPU LLM smoke: ngl=0 =="
"$LLAMA_BIN" "${common_args[@]}" -ngl 0 >"$CPU_LOG" 2>&1
tail -40 "$CPU_LOG"

echo
echo "== fake-CUDA offload LLM smoke: ngl=$GPU_LAYERS =="
if [[ "${LANXIN_LLM_TRACE:-0}" != "0" ]]; then
  LANXIN_NVIDIA_CUDA_TRACE=1 LANXIN_NVIDIA_CUDA_WAIT_COMPLETION=1 \
    "$LLAMA_BIN" "${common_args[@]}" -ngl "$GPU_LAYERS" >"$CUDA_LOG" 2>&1
else
  "$LLAMA_BIN" "${common_args[@]}" -ngl "$GPU_LAYERS" >"$CUDA_LOG" 2>&1
fi
tail -40 "$CUDA_LOG"

echo
echo "== summary =="
echo "cpu_log=$CPU_LOG"
echo "cuda_log=$CUDA_LOG"
if [[ "${LANXIN_LLM_TRACE:-0}" != "0" ]]; then
  echo "cuda_module_loads=$(grep -c 'cuModuleLoadData' "$CUDA_LOG" 2>/dev/null || true)"
  echo "cuda_launch_scaffolds=$(grep -c 'cuLaunchKernel scaffold' "$CUDA_LOG" 2>/dev/null || true)"
else
  echo "cuda_trace=disabled; rerun with LANXIN_LLM_TRACE=1 to count CUDA module loads and launch scaffolds"
fi
echo "note: this demonstrates llama.cpp CUDA API/code-load/offload plumbing. Real GPU math still depends on QMD release completion being implemented."
