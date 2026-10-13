from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

import yaml

from modules.metrics.gpu import _detect_all_gpu_ids, _largest_power_of_two_leq

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0")
_DEFAULT_VLLM_PORT = 8001
# Default vLLM venv (Qwen-compatible transformers). A client can override it
# with `vllm.vllm_bin` to run from a separate env — e.g. GLM-4.6V needs a newer
# transformers than the coder, so it points at /opt/vllm-glm-env/bin/vllm.
_DEFAULT_VLLM_BIN = "/opt/vllm-env/bin/vllm"
_DEFAULT_REASONING_PARSER = "qwen3"
_AUTO_GPU_TOKENS = {"", "auto", "all"}


@dataclass(frozen=True)
class VllmJob:
    name: str
    model: str
    revision: str | None
    port: int
    gpu_ids: str
    tp: int
    gpu_util: float
    max_len: int
    max_seqs: int
    api_key: str
    vllm_bin: str
    trust_remote_code: bool
    reasoning_parser: str | None
    extra_args: tuple[str, ...]
    dp: int = 1


@dataclass
class _RawSpec:
    """Per-client spec before cross-client GPU allocation."""
    name: str
    model: str
    revision: str | None
    port: int
    gpu_util: float
    max_len: int
    max_seqs: int
    api_key: str
    explicit_ids: list[str] | None
    explicit_tp: int | None
    vllm_bin: str
    trust_remote_code: bool
    reasoning_parser: str | None
    extra_args: tuple[str, ...]
    data_parallel: int = 1

# Check if the URL is a local host
def _is_local(url: str | None) -> bool:
    u = (url or "").lower()
    return any(h in u for h in _LOCAL_HOSTS)


# Get the port from the base URL
def _port_from_base_url(base_url: str | None, override: Any) -> int:
    if override is not None:
        return int(override)
    m = re.search(r":(\d+)(?:/|$)", base_url or "")
    return int(m.group(1)) if m else _DEFAULT_VLLM_PORT


# Parse the explicit GPU IDs from the YAML
def _parse_explicit_ids(raw: Any) -> list[str] | None:
    """Return list of GPU IDs from YAML, or None if auto/missing."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in _AUTO_GPU_TOKENS:
        return None
    return [x.strip() for x in str(raw).split(",") if x.strip()]

# Parse the explicit tensor parallel size from the YAML
def _parse_explicit_tp(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


# Resolve the reasoning parser: absent -> default ("qwen3"); explicit null/empty -> None (omit flag).
def _parse_reasoning_parser(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None

# Collect the raw specifications from the YAML
def _collect_raw_specs(cfg: dict[str, Any]) -> list[_RawSpec]:
    llm = cfg.get("llm_clients") or {}
    specs: list[_RawSpec] = []
    for name, spec in llm.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("enabled", True) is False:
            continue
        v = spec.get("vllm") or {}
        if not isinstance(v, dict):
            continue
        model = (v.get("model") or "").strip()
        if not model:
            continue
        base = spec.get("base_url") or ""
        if not _is_local(base):
            continue

        specs.append(_RawSpec(
            name=name,
            model=model,
            revision=(str(v.get("revision")).strip() or None) if v.get("revision") is not None else None,
            port=_port_from_base_url(base, v.get("port")),
            gpu_util=float(v.get("gpu_memory_utilization", 0.90)),
            max_len=int(v.get("max_model_len", 8192)),
            max_seqs=int(v.get("max_num_seqs", 4)),
            api_key=str(v.get("api_key", "local")),
            explicit_ids=_parse_explicit_ids(v.get("gpu_ids")),
            explicit_tp=_parse_explicit_tp(v.get("tensor_parallel_size")),
            vllm_bin=str(v.get("vllm_bin") or _DEFAULT_VLLM_BIN),
            trust_remote_code=bool(v.get("trust_remote_code", False)),
            reasoning_parser=_parse_reasoning_parser(
                v.get("reasoning_parser", _DEFAULT_REASONING_PARSER)
            ),
            extra_args=tuple(str(x) for x in (v.get("extra_args") or [])),
            data_parallel=int(v.get("data_parallel_size", 1) or 1),
        ))
    return specs


# Allocate the GPUs to the specifications
def _allocate_gpus(specs: list[_RawSpec], all_gpus: list[str]) -> dict[str, list[str]]:
    """
    Input:
        specs: list of raw specifications
        all_gpus: list of all visible GPUs
    Output:
        assigned: dictionary of client names to list of GPU IDs
    """
    assigned: dict[str, list[str]] = {}
    used: set[str] = set()

    # Phase 1: explicit gpu_ids
    for s in specs:
        if s.explicit_ids is None:
            continue
        for g in s.explicit_ids:
            if g not in all_gpus:
                raise ValueError(
                    f"{s.name}: gpu_ids includes {g!r} but visible GPUs are {all_gpus}"
                )
            if g in used:
                raise ValueError(
                    f"{s.name}: GPU {g} already reserved by another client"
                )
            used.add(g)
        assigned[s.name] = list(s.explicit_ids)

    # Phase 2: auto gpu_ids + explicit tp
    free = [g for g in all_gpus if g not in used]
    for s in specs:
        if s.explicit_ids is not None:
            continue
        if s.explicit_tp is None:
            continue
        if len(free) < s.explicit_tp:
            raise ValueError(
                f"{s.name}: tensor_parallel_size={s.explicit_tp} but only "
                f"{len(free)} GPU(s) free after explicit reservations"
            )
        assigned[s.name] = free[: s.explicit_tp]
        free = free[s.explicit_tp:]

    # Phase 3: fully auto — split remaining evenly
    auto_specs = [s for s in specs if s.name not in assigned]
    if auto_specs:
        if len(free) < len(auto_specs):
            raise ValueError(
                f"{len(auto_specs)} auto client(s) but only {len(free)} GPU(s) "
                f"free — specify gpu_ids explicitly or remove a client"
            )
        per = len(free) // len(auto_specs)
        remainder = len(free) % len(auto_specs)
        idx = 0
        for i, s in enumerate(auto_specs):
            n = per + (1 if i < remainder else 0)
            assigned[s.name] = free[idx: idx + n]
            idx += n

    return assigned


def _finalize_jobs(specs: list[_RawSpec], assigned: dict[str, list[str]]) -> list[VllmJob]:
    """Decide the tensor parallel size for each job
    Input:
        specs: list of raw specifications
        assigned: dictionary of client names to list of GPU IDs
    Output:
        jobs: list of vLLM jobs
    """
    pow2 = os.environ.get("VLLM_TP_POWER_OF_TWO", "").strip() in ("1", "true", "yes")
    jobs: list[VllmJob] = []
    for s in specs:
        ids = assigned[s.name]
        # With data parallelism the client owns tp * dp GPUs: dp independent
        # replicas, each sharded over tp cards.
        dp = max(1, s.data_parallel)
        if s.explicit_tp is not None:
            if s.explicit_tp * dp != len(ids):
                raise ValueError(
                    f"{s.name}: tensor_parallel_size={s.explicit_tp} * "
                    f"data_parallel_size={dp} = {s.explicit_tp * dp} but "
                    f"{len(ids)} GPU(s) assigned ({ids})"
                )
            tp = s.explicit_tp
        else:
            tp = max(1, len(ids) // dp)
            if pow2:
                tp = _largest_power_of_two_leq(tp)
        jobs.append(VllmJob(
            name=s.name, model=s.model, revision=s.revision, port=s.port,
            gpu_ids=",".join(ids), tp=tp, dp=dp,
            gpu_util=s.gpu_util, max_len=s.max_len,
            max_seqs=s.max_seqs, api_key=s.api_key,
            vllm_bin=s.vllm_bin, trust_remote_code=s.trust_remote_code,
            reasoning_parser=s.reasoning_parser, extra_args=s.extra_args,
        ))
    return jobs


def _build_jobs(cfg: dict[str, Any]) -> list[VllmJob]:
    """Build the vLLM jobs from the configuration
    Input:
        cfg: dictionary of configuration
    Output:
        jobs: list of vLLM jobs
    """
    # Collect the raw specifications from the YAML
    specs = _collect_raw_specs(cfg)
    if not specs:
        return []
    # Detect all visible GPUs
    all_gpus = _detect_all_gpu_ids()
    # Allocate the GPUs to the specifications
    assigned = _allocate_gpus(specs, all_gpus)
    # Decide the tensor parallel size for each job
    jobs = _finalize_jobs(specs, assigned)

    print(
        f"[vllm-spawn] Visible GPUs: {all_gpus} | Local Clients: "
        f"{[s.name for s in specs]}",
        flush=True,
    )
    return jobs


def _build_cmd(job: VllmJob) -> list[str]:
    cmd = [
        job.vllm_bin, "serve", job.model,
        "--port", str(job.port),
        "--api-key", job.api_key,
        "--max-model-len", str(job.max_len),
        "--tensor-parallel-size", str(job.tp),
        *(["--data-parallel-size", str(job.dp)] if job.dp > 1 else []),
        "--gpu-memory-utilization", str(job.gpu_util),
        "--max_num_seqs", str(job.max_seqs),
        "--generation-config", "vllm",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--max-num-batched-tokens", "8192",
    ]
    if job.revision:
        cmd += ["--revision", job.revision, "--tokenizer-revision", job.revision]
    if job.reasoning_parser:
        cmd += ["--reasoning-parser", job.reasoning_parser]
    if job.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd += list(job.extra_args)
    return cmd


def _bin_exists(path: str) -> bool:
    """True if the vLLM binary is runnable (absolute/relative path or on PATH)."""
    if "/" in path:
        return os.path.isfile(path) and os.access(path, os.X_OK)
    return shutil.which(path) is not None


_READY_TIMEOUT_ENV = "VLLM_ENGINE_READY_TIMEOUT_S"
_DEFAULT_READY_TIMEOUT_S = "3600"
_SUPERVISE_S = int(os.environ.get("VLLM_SPAWN_SUPERVISE_S", "2700"))
_MAX_RESTARTS = int(os.environ.get("VLLM_SPAWN_MAX_RESTARTS", "2"))


def _is_hf_repo(model: str) -> bool:
    return "/" in model and not os.path.exists(model)


def _prefetch_models(jobs: list[VllmJob]) -> None:
    """Download every distinct (model, revision) into the HF cache before any vLLM process starts, so engine start-up never
    waits on the network. Skipped when HF_HUB_OFFLINE is set; a failed download is logged and left to vLLM itself."""
    if os.environ.get("HF_HUB_OFFLINE", "").strip() in ("1", "true", "yes"):
        return
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # noqa: BLE001
        print(f"[vllm-spawn] prefetch skipped (huggingface_hub unavailable: {e})", flush=True)
        return
    targets = {(j.model, j.revision) for j in jobs if _is_hf_repo(j.model)}
    if not targets:
        return
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    results: dict[tuple[str, str | None], str] = {}

    def one(model: str, rev: str | None) -> None:
        t0 = time.time()
        try:
            path = snapshot_download(model, revision=rev, max_workers=8)
            results[(model, rev)] = f"ok {time.time() - t0:.0f}s -> {path}"
        except Exception as e:  # noqa: BLE001
            results[(model, rev)] = f"FAILED after {time.time() - t0:.0f}s: {type(e).__name__}: {str(e).splitlines()[0][:120]}"

    print(f"[vllm-spawn] prefetching {len(targets)} model(s): " + ", ".join(f"{m}@{r or 'main'}" for m, r in sorted(targets)), flush=True)
    threads = [threading.Thread(target=one, args=t, daemon=True) for t in sorted(targets)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for (m, r), msg in sorted(results.items()):
        print(f"[vllm-spawn] prefetch {m}@{r or 'main'}: {msg}", flush=True)


def _healthy(job: VllmJob) -> bool:
    req = urllib.request.Request(f"http://127.0.0.1:{job.port}/v1/models", headers={"Authorization": f"Bearer {job.api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def _spawn_one(job: VllmJob) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = job.gpu_ids
    env.setdefault(_READY_TIMEOUT_ENV, _DEFAULT_READY_TIMEOUT_S)
    cmd = _build_cmd(job)
    print(
        f"[vllm-spawn] Starting vLLM Client: {job.name} | Model: {job.model} | "
        f"Revision: {job.revision or 'main'} | Port: {job.port} | "
        f"GPUs: {job.gpu_ids} | Tensor Parallel Size: {job.tp} | "
        f"Data Parallel Size: {job.dp} | bin: {job.vllm_bin} | "
        f"reasoning_parser: {job.reasoning_parser or '-'} | trust_remote_code: {job.trust_remote_code}",
        flush=True,
    )
    return subprocess.Popen(cmd, env=env, start_new_session=True)


def main() -> int:
    path = os.environ.get("CONFIG_FILE", "/workspace/configuration.yaml")
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[vllm-spawn] Configuration file not found: {path}", file=sys.stderr)
        return 1
    except yaml.YAMLError as e:
        print(f"[vllm-spawn] Invalid YAML in {path}: {e}", file=sys.stderr)
        return 1

    try:
        jobs = _build_jobs(cfg)
    except ValueError as e:
        print(f"[vllm-spawn] Error: {e}", file=sys.stderr)
        return 1

    if not jobs:
        print("[vllm-spawn] No local vLLM clients configured — nothing to start", flush=True)
        return 0

    # Spawn each job independently: a missing binary or launch failure for one
    # client must NOT abort the others (e.g. GLM env not built yet should still
    # let the coder come up).
    runnable = [j for j in jobs if _bin_exists(j.vllm_bin)]
    _prefetch_models(runnable)
    failures = 0
    procs: dict[str, subprocess.Popen] = {}
    for j in jobs:
        if not _bin_exists(j.vllm_bin):
            print(
                f"[vllm-spawn] ERROR {j.name}: vLLM binary not found: {j.vllm_bin!r} — "
                f"build the env (e.g. `bash scripts/setup_glm_vllm_env.sh`) or fix "
                f"`vllm.vllm_bin` in the config. Skipping this client.",
                file=sys.stderr,
            )
            failures += 1
            continue
        try:
            procs[j.name] = _spawn_one(j)
        except Exception as e:  # noqa: BLE001 - one client's failure must not kill the rest
            print(f"[vllm-spawn] ERROR {j.name}: spawn failed: {e}", file=sys.stderr)
            failures += 1

    launched = len(jobs) - failures
    print(
        f"[vllm-spawn] {launched}/{len(jobs)} vLLM instances launched in background",
        flush=True,
    )
    _supervise({j.name: j for j in jobs if j.name in procs}, procs)
    return 1 if failures else 0


def _supervise(jobs: dict[str, VllmJob], procs: dict[str, subprocess.Popen]) -> None:
    """Until every instance answers /v1/models (or the budget runs out): restart an instance whose process exited before
    becoming healthy, at most _MAX_RESTARTS times each."""
    restarts = {n: 0 for n in procs}
    healthy: set[str] = set()
    t0 = time.time()
    while procs and len(healthy) < len(procs) and time.time() - t0 < _SUPERVISE_S:
        time.sleep(10)
        for name, p in list(procs.items()):
            if name in healthy:
                continue
            if _healthy(jobs[name]):
                healthy.add(name)
                print(f"[vllm-spawn] {name} healthy after {time.time() - t0:.0f}s", flush=True)
                continue
            rc = p.poll()
            if rc is None:
                continue
            if restarts[name] >= _MAX_RESTARTS:
                print(f"[vllm-spawn] ERROR {name}: exited with code {rc} and restart budget exhausted", file=sys.stderr, flush=True)
                healthy.add(name)  # stop watching it
                continue
            restarts[name] += 1
            print(f"[vllm-spawn] {name} exited with code {rc} before becoming healthy — restart {restarts[name]}/{_MAX_RESTARTS}", flush=True)
            try:
                procs[name] = _spawn_one(jobs[name])
            except Exception as e:  # noqa: BLE001
                print(f"[vllm-spawn] ERROR {name}: restart failed: {e}", file=sys.stderr, flush=True)
    pending = [n for n in procs if n not in healthy]
    if pending:
        print(f"[vllm-spawn] supervision ended after {time.time() - t0:.0f}s; not yet healthy: {pending}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
