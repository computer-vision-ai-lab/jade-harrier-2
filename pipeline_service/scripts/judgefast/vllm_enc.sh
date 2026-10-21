#!/usr/bin/env bash
# Judge vllm_bin wrapper ("enc" patch): the GLM env's vllm/config/scheduler.py is patched by run.sh
# (scripts/judgefast/make_sched.py) so the encoder cache budget comes from $VLLM_ENC_CACHE_TOKENS instead of max_num_batched_tokens;
# 1.5 M tokens keeps the vision encodings of a whole judge batch resident (the default budget = 16384 re-encodes images every step).
# With the variable unset the patched scheduler behaves exactly like the original.
export VLLM_ENC_CACHE_TOKENS=${JF_ENC_TOKENS:-1500000}
GLM_VLLM=${GLM_VLLM_BIN:-/opt/vllm-glm-env/bin/vllm}
export PATH=$(dirname "$GLM_VLLM"):$PATH
echo "[judgefast] VLLM_ENC_CACHE_TOKENS=$VLLM_ENC_CACHE_TOKENS -> $GLM_VLLM $*" >&2
exec "$GLM_VLLM" "$@"
