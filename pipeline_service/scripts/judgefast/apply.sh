#!/usr/bin/env bash
# apply.sh [glm-venv]: patch the GLM env's vllm/config/scheduler.py (encoder cache budget from $VLLM_ENC_CACHE_TOKENS). Idempotent, md5-checked
# (make_sched.py refuses an unexpected vLLM); exit 0 = patched or already patched, exit 1 = vLLM differs (the judges then run unpatched = stock).
set -u; V=${1:-/opt/vllm-glm-env}; D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
F=$(ls $V/lib/python3.*/site-packages/vllm/config/scheduler.py 2>/dev/null | head -1); [ -n "$F" ] || { echo "[judgefast] no scheduler.py under $V"; exit 1; }
OUT_MD5=0f758a80715c6e042af17b0ae36c887e
if [ "$(md5sum "$F" | cut -d' ' -f1)" = "$OUT_MD5" ]; then echo "[judgefast] already patched: $F"; exit 0; fi
python3 "$D/make_sched.py" "$F" "$F.patched" && cp "$F.patched" "$F" && rm -f "$F.patched" && find "$(dirname "$F")" -name "scheduler.*.pyc" -delete && echo "[judgefast] patched $F"
