from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------
# Helpers
# ---------------------------

def ts_de() -> str:
    # DD-MM-YYYY_H-M-S
    return dt.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


def iso_now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def shquote(s: str) -> str:
    """Safe single-quote for bash -lc."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_capture(cmd: List[str], *, check: bool = True) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, text=True, capture_output=True)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    return p.returncode, p.stdout, p.stderr


def run(cmd: List[str], *, check: bool = True) -> None:
    code, out, err = run_capture(cmd, check=check)
    _ = (code, out, err)


def cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# ---------------------------
# HTTP (LM Studio)
# ---------------------------

def http_post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int = 600) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} for {url}: {body}") from e


def http_get(url: str, headers: Dict[str, str], timeout: int = 60) -> Dict[str, Any]:
    req = urllib.request.Request(url, method="GET", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def lm_headers(cfg: Dict[str, Any]) -> Dict[str, str]:
    # Optional API token (Bearer). Setze in config:
    # lmstudio.api_token_env = "LM_API_TOKEN"
    env_name = cfg_get(cfg, "lmstudio.api_token_env", "").strip()
    token = os.environ.get(env_name, "").strip() if env_name else ""
    if token:
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    return {"Content-Type": "application/json"}


def lm_unload(cfg: Dict[str, Any]) -> None:
    base = cfg["lmstudio"]["base_url"].rstrip("/")
    model_key = cfg["lmstudio"]["model"]
    headers = lm_headers(cfg)
    url = f"{base}/api/v1/models/unload"
    # Best-effort unload
    try:
        http_post_json(url, {"instance_id": model_key}, headers=headers, timeout=120)
    except Exception:
        pass


def lm_load(cfg: Dict[str, Any], kv_on: bool) -> Dict[str, Any]:
    base = cfg["lmstudio"]["base_url"].rstrip("/")
    model_key = cfg["lmstudio"]["model"]
    headers = lm_headers(cfg)
    url = f"{base}/api/v1/models/load"

    common = dict(cfg["lmstudio"].get("load_config_common", {}))
    payload = {"model": model_key, **common}
    payload["offload_kv_cache_to_gpu"] = bool(kv_on)

    return http_post_json(url, payload, headers=headers, timeout=600)


def lm_reload_for_condition(cfg: Dict[str, Any], kv_on: bool) -> Dict[str, Any]:
    # Unload -> Load (damit KV-Cache-Setting sicher applied)
    lm_unload(cfg)
    resp = lm_load(cfg, kv_on=kv_on)
    # kurze Pause (optional)
    time.sleep(1.0)
    return resp


# ---------------------------
# SSH / rsync / scp
# ---------------------------

def ssh_cmd(cfg: Dict[str, Any], user: str, host: str, remote_bash: str) -> List[str]:
    opts = cfg_get(cfg, "orchestrator.ssh_options", [])
    return ["ssh", *opts, f"{user}@{host}", "--", "bash", "-lc", remote_bash]


def rsync_pull(cfg: Dict[str, Any], user: str, host: str, remote_dir: str, local_dir: Path) -> None:
    ensure_dir(local_dir)
    opts = cfg_get(cfg, "orchestrator.ssh_options", [])
    ssh = ["ssh", *opts]
    cmd = [
        "rsync", "-a",
        "-e", " ".join(ssh),
        f"{user}@{host}:{remote_dir.rstrip('/')}/",
        str(local_dir),
    ]
    run(cmd, check=True)


def scp_pull(cfg: Dict[str, Any], user: str, host: str, remote_file: str, local_file: Path) -> None:
    ensure_dir(local_file.parent)
    opts = cfg_get(cfg, "orchestrator.ssh_options", [])
    cmd = ["scp", *opts, f"{user}@{host}:{remote_file}", str(local_file)]
    run(cmd, check=True)


# ---------------------------
# AgentDojo parsing
# ---------------------------

def parse_suite_blocks(terminal_log: Path) -> Dict[str, Any]:
    """
    Parsed pro Suite (workspace, travel, banking, slack, combined):
      Average utility: 42.47%
      Passed injection tasks as user tasks: 21/35
      Average security: 28.35%
    """
    out: Dict[str, Any] = {"suites": {}}
    if not terminal_log.exists():
        return out

    current: Optional[str] = None
    for line in terminal_log.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith("Results for suite "):
            current = s.split("Results for suite ", 1)[1].strip()
            out["suites"][current] = {}
            continue
        if current is None:
            continue

        if s.startswith("Average utility:"):
            val = s.split("Average utility:", 1)[1].strip().rstrip("%")
            try:
                out["suites"][current]["utility_pct"] = float(val)
            except ValueError:
                pass
        elif s.startswith("Average security:"):
            val = s.split("Average security:", 1)[1].strip().rstrip("%")
            try:
                out["suites"][current]["security_pct"] = float(val)
            except ValueError:
                pass
        elif s.startswith("Passed injection tasks as user tasks:"):
            frac = s.split(":", 1)[1].strip()
            out["suites"][current]["passed_injection_as_user_tasks"] = frac

    if "combined" in out["suites"]:
        out["combined"] = out["suites"]["combined"]
    return out


# ---------------------------
# Core orchestration
# ---------------------------

def build_agentdojo_remote_cmd(cfg: Dict[str, Any], remote_run_dir: str) -> str:
    ad = cfg["agentdojo"]
    py = ad["python"]
    wd = ad["workdir"]
    benchmark_version = ad["benchmark_version"]
    attack = ad["attack"]
    model_wrapper = ad["model_wrapper"]
    model_id = ad["model_id"]

    extra_args = ad.get("extra_args", [])
    extra = " ".join(shquote(str(x)) for x in extra_args)

    env = ad.get("env", {})
    env_prefix = " ".join([f"{k}={shquote(str(v))}" for k, v in env.items()])

    cmd = (
        f"cd {shquote(wd)} && "
        f"mkdir -p {shquote(remote_run_dir)} && "
        f"{env_prefix} "
        f"{shquote(py)} -m agentdojo.scripts.benchmark "
        f"--model {shquote(model_wrapper)} "
        f"--model-id {shquote(model_id)} "
        f"--logdir {shquote(remote_run_dir)} "
        f"--benchmark-version {shquote(benchmark_version)} "
        f"--attack {shquote(attack)} "
        f"{extra} "
        f"2>&1 | tee {shquote(remote_run_dir)}/terminal.log"
    )
    return cmd


def snapshot_lmstudio_logs(
    cfg: Dict[str, Any],
    run_id: str,
    marker_path: str,
    local_run_dir: Path,
) -> Optional[Path]:
    """
    Packs LM Studio server logs newer than marker_path into tar.gz and pulls it.
    Optional cleanup:
      lmstudio.log_cleanup_mode: "none" | "move" | "delete"
      lmstudio.server_logs_archive_dir: path (if move)
    """
    lm = cfg["lmstudio"]
    lm_user = lm["ssh_user"]
    lm_host = lm["ssh_host"]
    logs_dir = lm["server_logs_dir"]

    cleanup_mode = lm.get("log_cleanup_mode", "none")
    archive_dir = lm.get("server_logs_archive_dir", f"{logs_dir}-archive")

    remote_tar = f"/tmp/lmstudio_logs_{run_id}.tar.gz"
    remote_list = f"/tmp/lmstudio_files_{run_id}.txt"

    # Create tar of files newer than marker
    remote_script = (
        "set -euo pipefail; "
        f"test -f {shquote(marker_path)}; "
        f"rm -f {shquote(remote_tar)} {shquote(remote_list)}; "
        f"LOGROOT={shquote(logs_dir)}; "
        f"find \"$LOGROOT\" -type f -newer {shquote(marker_path)} -print > {shquote(remote_list)} || true; "
        f"if [ -s {shquote(remote_list)} ]; then "
        f"  tar -czf {shquote(remote_tar)} -T {shquote(remote_list)}; "
        f"else "
        f"  tar -czf {shquote(remote_tar)} --files-from /dev/null; "
        f"fi; "
        "echo OK"
    )
    run(ssh_cmd(cfg, lm_user, lm_host, remote_script), check=True)

    local_tar = local_run_dir / "lmstudio_logs.tar.gz"
    scp_pull(cfg, lm_user, lm_host, remote_tar, local_tar)

    # Cleanup on LM host
    if cleanup_mode == "delete":
        del_script = (
            "set -euo pipefail; "
            f"LOGROOT={shquote(logs_dir)}; "
            f"find \"$LOGROOT\" -type f -newer {shquote(marker_path)} -delete || true; "
            "echo OK"
        )
        run(ssh_cmd(cfg, lm_user, lm_host, del_script), check=True)
    elif cleanup_mode == "move":
        move_script = (
            "set -euo pipefail; "
            f"mkdir -p {shquote(archive_dir)}/{shquote(run_id)}; "
            f"LOGROOT={shquote(logs_dir)}; "
            f"find \"$LOGROOT\" -type f -newer {shquote(marker_path)} -print0 | "
            "while IFS= read -r -d '' f; do "
            f"  rel=${{f#\"$LOGROOT\"/}}; "
            f"  mkdir -p {shquote(archive_dir)}/{shquote(run_id)}/\"$(dirname \"$rel\")\"; "
            f"  mv -f \"$f\" {shquote(archive_dir)}/{shquote(run_id)}/\"$rel\"; "
            "done; "
            "echo OK"
        )
        run(ssh_cmd(cfg, lm_user, lm_host, move_script), check=True)

    # Remove temp files + marker
    final_script = (
        "set -euo pipefail; "
        f"rm -f {shquote(remote_tar)} {shquote(remote_list)} {shquote(marker_path)} || true; "
        "echo OK"
    )
    run(ssh_cmd(cfg, lm_user, lm_host, final_script), check=True)

    return local_tar


def append_results_row(results_csv: Path, row: Dict[str, Any]) -> None:
    ensure_dir(results_csv.parent)
    exists = results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def validate_cfg(cfg: Dict[str, Any]) -> None:
    required = [
        "study.dataset_slug",
        "study.pairs",
        "study.pair_order",
        "lmstudio.base_url",
        "lmstudio.model",
        "lmstudio.server_logs_dir",
        "lmstudio.ssh_user",
        "lmstudio.ssh_host",
        "agentdojo.ssh_user",
        "agentdojo.ssh_host",
        "agentdojo.workdir",
        "agentdojo.python",
        "agentdojo.attack",
        "agentdojo.benchmark_version",
        "agentdojo.model_wrapper",
        "agentdojo.model_id",
        "agentdojo.remote_runs_root",
        "orchestrator.local_runs_root",
    ]
    missing = [k for k in required if cfg_get(cfg, k, None) in (None, "", [])]
    if missing:
        raise SystemExit(f"Config missing/empty fields: {missing}")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: orchestrator.py bench_config.local.json")
        return 2

    cfg_path = Path(sys.argv[1]).resolve()
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return 2

    cfg = load_json(cfg_path)
    validate_cfg(cfg)

    # Dataset paths (local on orchestrator VM)
    local_runs_root = (Path(__file__).parent / cfg["orchestrator"]["local_runs_root"]).resolve()
    ensure_dir(local_runs_root)

    dataset_slug = cfg["study"]["dataset_slug"]
    dataset_dir = local_runs_root / dataset_slug
    ensure_dir(dataset_dir)

    results_csv = dataset_dir / "results.csv"
    manifest_path = dataset_dir / "runs_manifest.json"

    # Load/Init manifest
    manifest: Dict[str, Any]
    if manifest_path.exists():
        try:
            manifest = load_json(manifest_path)
        except Exception:
            manifest = {}
    else:
        manifest = {}

    if "dataset_slug" not in manifest:
        manifest = {
            "dataset_slug": dataset_slug,
            "generated_at": ts_de(),
            "release_asset": None,
            "release_sha256": None,
            "runs": []
        }

    pairs = int(cfg["study"]["pairs"])
    order = str(cfg["study"]["pair_order"]).strip().lower()
    if order not in ("on_then_off", "off_then_on"):
        raise SystemExit("study.pair_order must be 'on_then_off' or 'off_then_on'")

    # Hosts
    lm_user = cfg["lmstudio"]["ssh_user"]
    lm_host = cfg["lmstudio"]["ssh_host"]

    ad_user = cfg["agentdojo"]["ssh_user"]
    ad_host = cfg["agentdojo"]["ssh_host"]
    remote_runs_root = cfg["agentdojo"]["remote_runs_root"].rstrip("/")

    # Optional: sanity check LM API reachable (non-fatal)
    try:
        headers = lm_headers(cfg)
        _ = http_get(cfg["lmstudio"]["base_url"].rstrip("/") + "/api/v1/models", headers=headers, timeout=10)
    except Exception:
        pass

    # Pair execution
    for pair_idx in range(1, pairs + 1):
        conds = [True, False] if order == "on_then_off" else [False, True]

        for kv_on in conds:
            kv_label = "kv_on" if kv_on else "kv_off"
            run_id = f"pair{pair_idx:02d}_{kv_label}_{ts_de()}"
            run_dir = dataset_dir / run_id
            ensure_dir(run_dir)

            started_at = iso_now()
            t0 = time.time()

            # Marker on LM host BEFORE run (for log snapshot)
            marker = f"/tmp/lm_orch_marker_{run_id}"
            run(ssh_cmd(cfg, lm_user, lm_host, f"rm -f {shquote(marker)}; touch {shquote(marker)}"), check=True)

            # LM reload for condition
            lm_load_resp = lm_reload_for_condition(cfg, kv_on=kv_on)
            write_json(run_dir / "lmstudio_load_response.json", lm_load_resp)

            # AgentDojo remote directory
            remote_run_dir = f"{remote_runs_root}/{dataset_slug}/{run_id}"
            remote_cmd = build_agentdojo_remote_cmd(cfg, remote_run_dir)
            write_json(run_dir / "invocation.json", {
                "run_id": run_id,
                "pair_index": pair_idx,
                "kv_cache": "on" if kv_on else "off",
                "remote_run_dir": remote_run_dir,
                "remote_cmd": remote_cmd,
                "started_at": started_at,
            })

            # Execute benchmark remote
            run(ssh_cmd(cfg, ad_user, ad_host, remote_cmd), check=True)

            # Pull AgentDojo artifacts
            agentdojo_local = run_dir / "agentdojo"
            ensure_dir(agentdojo_local)
            rsync_pull(cfg, ad_user, ad_host, remote_run_dir, agentdojo_local)

            # Snapshot LM logs for this run
            lm_tar = snapshot_lmstudio_logs(cfg, run_id, marker, run_dir)

            ended_at = iso_now()
            duration_sec = int(time.time() - t0)

            # Parse terminal log
            terminal_log = agentdojo_local / "terminal.log"
            parsed = parse_suite_blocks(terminal_log)
            write_json(run_dir / "parsed_metrics.json", parsed)

            combined = parsed.get("combined", {})
            utility = combined.get("utility_pct")
            security = combined.get("security_pct")
            passed_frac = combined.get("passed_injection_as_user_tasks")

            # Optional derived ASR estimate (if you want it in CSV)
            asr_est = (100.0 - security) if isinstance(security, (int, float)) else None

            # Results CSV
            row = {
                "run_id": run_id,
                "pair_index": pair_idx,
                "kv_cache": "on" if kv_on else "off",
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_sec": duration_sec,
                "utility_avg_pct_combined": utility,
                "security_avg_pct_combined": security,
                "passed_injection_as_user_tasks_combined": passed_frac,
                "asr_targeted_est_pct": asr_est,
            }
            append_results_row(results_csv, row)

            # Update manifest (dataset-level index)
            run_entry: Dict[str, Any] = {
                "run_id": run_id,
                "pair_index": pair_idx,
                "kv_cache": "on" if kv_on else "off",
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_sec": duration_sec,
                "remote": {
                    "agentdojo_host": ad_host,
                    "agentdojo_logdir": remote_run_dir,
                },
                "lmstudio": {
                    "host": lm_host,
                    "base_url": cfg["lmstudio"]["base_url"],
                    "model": cfg["lmstudio"]["model"],
                    "offload_kv_cache_to_gpu": kv_on,
                    "eval_batch_size": cfg_get(cfg, "lmstudio.load_config_common.eval_batch_size"),
                    "context_length": cfg_get(cfg, "lmstudio.load_config_common.context_length"),
                },
                "artifacts": {
                    "agentdojo_terminal_log": "agentdojo/terminal.log" if terminal_log.exists() else None,
                    "lmstudio_logs_tar_gz": "lmstudio_logs.tar.gz" if lm_tar and lm_tar.exists() else None,
                },
                "checksums": {},
                "metrics": {
                    "combined": {
                        "utility_pct": utility,
                        "security_pct": security,
                        "passed_injection_as_user_tasks": passed_frac,
                        "asr_targeted_est_pct": asr_est,
                    }
                }
            }

            # Add checksums for key artifacts (optional but hilfreich)
            if terminal_log.exists():
                run_entry["checksums"]["agentdojo_terminal_log_sha256"] = sha256_file(terminal_log)
            if lm_tar and lm_tar.exists():
                run_entry["checksums"]["lmstudio_logs_tar_gz_sha256"] = sha256_file(lm_tar)

            manifest["generated_at"] = ts_de()
            manifest.setdefault("runs", []).append(run_entry)
            write_json(manifest_path, manifest)

            print(f"[OK] {run_id} ({'KV_ON' if kv_on else 'KV_OFF'}) duration={duration_sec}s")

    print("\nDone.")
    print(f"Dataset dir: {dataset_dir}")
    print(f"- results.csv: {results_csv}")
    print(f"- runs_manifest.json: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())