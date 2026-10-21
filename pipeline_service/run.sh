#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_FILE="${CONFIG_PATH:-/workspace/configuration.yaml}"
export CONFIG_FILE

# Raise the open-file limit: the service runs hundreds of threads + Chromium sidecars; a 1024 soft limit
# breaks the validator ("Too many open files") and resets /status connections under load.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65536 2>/dev/null || true


# Thread pools: the GLM vLLM's CPU-side image preprocessing spawns nproc-sized OpenMP pools; on a small-quota
# host (10 CPUs seen on RunPod) that oversubscription made the judge's cold path 1.9x slower, while OMP=1 costs
# nothing on 20-40 CPU quotas. Inherited by every vLLM server spawned from llm/spawn.py.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# vLLM 0.32 / torch 2.13 coder: no static Triton launcher, no reuse of AOT compile artifacts across starts
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER="${TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER:-0}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"

# Judge-side settings (see scripts/burst/): JUDGE_IMG_UUID=1 sends each judge image
# once and then by uuid (llm/mm_resend.py, needs vLLM >= 0.23 on the judges); JUDGE_LB_MIGRATE=1 moves a task's judge
# affinity to the least-loaded replica (llm/balancer.py); BURST_GLM_URLS = GLM judges sleeping on the coder GPUs that
# scripts/burst/vllm_shim.sh starts (ports 8031.., one per coder GPU) when the coder's extra_args carry --burst-glm, swapped
# in for the judge-only tail of every batch by llm/burst.py through the coder proxy (BURST_PROXY = the coder base URL).
# Empty BURST_GLM_URLS = burst off (plain pass-through to vLLM).
export JUDGE_IMG_UUID="${JUDGE_IMG_UUID:-1}"
export JUDGE_LB_MIGRATE="${JUDGE_LB_MIGRATE:-1}"
export BURST_PROXY="${BURST_PROXY:-http://localhost:8001}"
if [ -z "${BURST_GLM_URLS+x}" ]; then
    BURST_GLM_URLS="$(python - <<'PYCFG'
import os, yaml
try:
    cfg = yaml.safe_load(open(os.environ["CONFIG_FILE"])) or {}
    for spec in (cfg.get("llm_clients") or {}).values():
        v = (spec or {}).get("vllm") or {}
        args = [str(x) for x in (v.get("extra_args") or [])]
        if "--burst-glm" in args and str(v.get("vllm_bin") or "").endswith("vllm_shim.sh"):
            n = len([g for g in str(v.get("gpu_ids") or "0").split(",") if g.strip()])
            print(",".join(f"http://localhost:{8031 + i}/v1" for i in range(n)))
            break
except Exception:
    pass
PYCFG
)"
    export BURST_GLM_URLS
fi
echo "[run.sh] judge settings: JUDGE_IMG_UUID=$JUDGE_IMG_UUID JUDGE_LB_MIGRATE=$JUDGE_LB_MIGRATE BURST_GLM_URLS=${BURST_GLM_URLS:-<off>}"

# Preflight: network, GPU count, per-GPU bf16 TFLOPS + decode-shaped weight-stream GB/s + power limit.
# Writes /tmp/preflight_metrics.json (picked up by serve.py for the miner-diag header).
if python -m modules.metrics.preflight; then
    echo "=== PRE-FLIGHT OK ==="
else
    echo "=== PRE-FLIGHT FAILED — starting FastAPI in REPLACE mode ==="
fi


# FastAPI
echo "=== STAGE 2: FastAPI ==="
python serve.py &
SERVE_PID=$!


# Dedicated GLM vLLM env (GLM-4.6V needs a newer transformers than the coder).
# Idempotent: only builds when the binary is missing. Non-fatal: on failure the
# judge/critic client is skipped and the coder still comes up.
GLM_VLLM_BIN="${GLM_VLLM_BIN:-/opt/vllm-glm-env/bin/vllm}"
if [ ! -x "$GLM_VLLM_BIN" ]; then
    echo "=== STAGE 2.5: building GLM vLLM env ($GLM_VLLM_BIN missing) ==="
    GLM_MODEL_INFO="$(GLM_VLLM_BIN="$GLM_VLLM_BIN" python - <<'PY'
import os, yaml
try:
    cfg = yaml.safe_load(open(os.environ["CONFIG_FILE"])) or {}
except Exception:
    raise SystemExit(0)
for spec in (cfg.get("llm_clients") or {}).values():
    if not isinstance(spec, dict):
        continue
    v = spec.get("vllm") or {}
    if str(v.get("vllm_bin") or "") == os.environ["GLM_VLLM_BIN"] and str(v.get("model") or "").strip():
        print(str(v["model"]).strip())
        print(str(v.get("revision") or "").strip())
        break
PY
)"
    GLM_MODEL="$(sed -n 1p <<<"$GLM_MODEL_INFO")"
    GLM_REVISION="$(sed -n 2p <<<"$GLM_MODEL_INFO")"
    [ -n "$GLM_MODEL" ] && echo "[run.sh] GLM env build target: $GLM_MODEL @ ${GLM_REVISION:-main}"
    MODEL="$GLM_MODEL" MODEL_REVISION="$GLM_REVISION" \
        bash "$SCRIPT_DIR/scripts/setup_glm_vllm_env.sh" \
        || echo "[run.sh] GLM env setup failed — judge/critic vLLM skipped, coder continues" >&2
else
    echo "[run.sh] GLM vLLM env present at $GLM_VLLM_BIN — skipping build"
fi
# Judge "enc" patch (scripts/judgefast): encoder cache budget from $VLLM_ENC_CACHE_TOKENS; the judges' vllm_bin (vllm_enc.sh) sets it.
[ -x "$GLM_VLLM_BIN" ] && bash "$SCRIPT_DIR/scripts/judgefast/apply.sh" "$(dirname "$(dirname "$GLM_VLLM_BIN")")" || echo "[run.sh] judgefast patch not applied (judges run stock)"


# Dedicated coder vLLM env. Same idea as the GLM env above: config-driven and
# idempotent, so it only builds when configuration.yaml points a client at
# $CODER_VLLM_BIN and the binary is missing. Non-fatal: failures are logged.
CODER_VLLM_BIN="${CODER_VLLM_BIN:-/opt/vllm-coder-env/bin/vllm}"
CODER_VLLM_VERSION="${CODER_VLLM_VERSION:-0.32.0}"
if [ ! -x "$CODER_VLLM_BIN" ]; then
    CODER_MODEL_INFO="$(GLM_VLLM_BIN="$CODER_VLLM_BIN" python - <<'PY'
import os, yaml
try:
    cfg = yaml.safe_load(open(os.environ["CONFIG_FILE"])) or {}
except Exception:
    raise SystemExit(0)
for spec in (cfg.get("llm_clients") or {}).values():
    if not isinstance(spec, dict):
        continue
    v = spec.get("vllm") or {}
    if str(v.get("vllm_bin") or "") == os.environ["GLM_VLLM_BIN"] and str(v.get("model") or "").strip():
        print(str(v["model"]).strip())
        print(str(v.get("revision") or "").strip())
        break
PY
)"
    CODER_MODEL="$(sed -n 1p <<<"$CODER_MODEL_INFO")"
    CODER_REVISION="$(sed -n 2p <<<"$CODER_MODEL_INFO")"
    if [ -n "$CODER_MODEL" ]; then
        echo "=== STAGE 2.6: building coder vLLM env $CODER_VLLM_VERSION ($CODER_VLLM_BIN missing) ==="
        echo "[run.sh] coder env build target: $CODER_MODEL @ ${CODER_REVISION:-main}"
        VENV="$(dirname "$(dirname "$CODER_VLLM_BIN")")" \
        VLLM_VERSION="$CODER_VLLM_VERSION" \
        MODEL="$CODER_MODEL" MODEL_REVISION="$CODER_REVISION" \
            bash "$SCRIPT_DIR/scripts/setup_glm_vllm_env.sh" \
            || echo "[run.sh] coder env setup failed — the coder cannot start without $CODER_VLLM_BIN; check logs" >&2
    fi
else
    echo "[run.sh] coder vLLM env present at $CODER_VLLM_BIN — skipping build"
fi


# vLLM spawn

echo "=== STAGE 3: vLLM spawn ==="
python -m llm.spawn || echo "[run.sh] vllm spawn returned non-zero — FastAPI continues for diagnostics" >&2

wait $SERVE_PID
