from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================
# Globals for monkeypatch state
# ============================================================

_PATCH_INSTALLED = False
_ORIGINAL_FROM_CONFIG_FUNC = None  # unbound function (cls, config) -> pipeline
_ACTIVE_SELECTION: Dict[str, Any] = {}
_ACTIVE_DEFENSE_IDS: List[str] = []
_ACTIVE_LOGDIR: Optional[Path] = None

# ============================================================
# Small utilities
# ============================================================

def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if callable(obj):
        return f"<callable:{getattr(obj, '__name__', obj.__class__.__name__)}>"
    return repr(obj)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(_json_safe(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")



def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _stdout(msg: str) -> None:
    print(msg)


# ============================================================
# Defense selection loading / normalization
# ============================================================

def _load_build_defense_selection() -> Callable[[List[str]], Any]:
    """
    Expected to exist in your local package:
      agentdojo_custom_defenses/custom_pipeline_factory.py
    """
    from agentdojo_custom_defenses.custom_pipeline_factory import build_defense_selection
    return build_defense_selection


def _normalize_selection(selection_raw: Any) -> Dict[str, Any]:
    """
    Make the selection object tolerant against different return styles from your factory:
    - dict
    - object with attributes
    """
    if isinstance(selection_raw, dict):
        sel = dict(selection_raw)
    else:
        sel = {"_selection_object": selection_raw}

        # common attributes (if factory returns a dataclass/object)
        for name in (
            "metadata",
            "tool_output_formatter",
            "output_formatter",
            "formatter",
            "call_gate",
            "tool_call_gate",
            "pre_tool_call_gate",
            "pre_tool_call",
            "d4_gate",
            "disclosure_gate",
            "apply_to_pipeline",
            "apply",
        ):
            if hasattr(selection_raw, name):
                try:
                    sel[name] = getattr(selection_raw, name)
                except Exception:
                    pass

    # normalize nested metadata
    meta = sel.get("metadata")
    if meta is None:
        # some factories store top-level fields only
        meta = {}
    elif not isinstance(meta, dict):
        meta = {"value": repr(meta)}
    sel["metadata"] = meta

    return sel


def _extract_formatter(sel: Dict[str, Any]) -> Optional[Callable[..., Any]]:
    # Most likely names first
    for key in ("tool_output_formatter", "output_formatter", "formatter"):
        fn = sel.get(key)
        if callable(fn):
            return fn

    # nested hooks
    hooks = sel.get("hooks")
    if isinstance(hooks, dict):
        for key in ("tool_output_formatter", "output_formatter", "formatter"):
            fn = hooks.get(key)
            if callable(fn):
                return fn

    return None


def _extract_call_gate(sel: Dict[str, Any]) -> Optional[Callable[..., Any]]:
    for key in ("call_gate", "tool_call_gate", "pre_tool_call_gate", "pre_tool_call"):
        fn = sel.get(key)
        if callable(fn):
            return fn

    hooks = sel.get("hooks")
    if isinstance(hooks, dict):
        for key in ("call_gate", "tool_call_gate", "pre_tool_call_gate", "pre_tool_call"):
            fn = hooks.get(key)
            if callable(fn):
                return fn

    return None


def _extract_apply_to_pipeline(sel: Dict[str, Any]) -> Optional[Callable[..., Any]]:
    for key in ("apply_to_pipeline", "apply"):
        fn = sel.get(key)
        if callable(fn):
            return fn
    return None


# ============================================================
# Object graph helpers (best-effort introspection)
# ============================================================

_SIMPLE_TYPES = (str, bytes, bytearray, int, float, bool, type(None))


def _iter_children(obj: Any) -> Iterable[Any]:
    if isinstance(obj, _SIMPLE_TYPES):
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield v
        return
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield v
        return

    try:
        d = vars(obj)
    except Exception:
        return

    for v in d.values():
        yield v


def _walk_objects(root: Any, *, max_depth: int = 6) -> Iterable[Any]:
    seen: set[int] = set()
    stack: List[Tuple[Any, int]] = [(root, 0)]

    while stack:
        obj, depth = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        yield obj

        if depth >= max_depth:
            continue

        try:
            for child in _iter_children(obj):
                if not isinstance(child, _SIMPLE_TYPES):
                    stack.append((child, depth + 1))
        except Exception:
            continue


def _find_by_class_name(root: Any, class_name: str, *, max_depth: int = 6) -> List[Any]:
    matches: List[Any] = []
    for obj in _walk_objects(root, max_depth=max_depth):
        try:
            if obj.__class__.__name__ == class_name:
                matches.append(obj)
        except Exception:
            continue
    return matches


# ============================================================
# Patching logic for D1/D2 (formatter) and D3 (call gate)
# ============================================================

def _set_tool_output_formatter_on_executor(executor: Any, formatter: Callable[..., Any]) -> bool:
    """
    Best-effort: AgentDojo versions may use different attribute names.
    """
    for attr in ("tool_output_formatter", "output_formatter", "formatter"):
        if hasattr(executor, attr):
            try:
                setattr(executor, attr, formatter)
                return True
            except Exception:
                pass
    return False


def _interpret_gate_decision(decision: Any) -> Tuple[bool, Any]:
    """
    Returns (allow, replacement_or_none)
    Supported patterns:
      - None -> allow
      - True/False
      - (allow_bool, replacement)
      - {"allow": bool, "return_value": ...}
      - {"decision": "allow"|"deny", "return_value": ...}
    """
    if decision is None:
        return True, None

    if isinstance(decision, bool):
        return decision, None

    if isinstance(decision, tuple) and len(decision) == 2 and isinstance(decision[0], bool):
        return decision[0], decision[1]

    if isinstance(decision, dict):
        if "allow" in decision:
            allow = bool(decision["allow"])
            return allow, decision.get("return_value")
        if "decision" in decision:
            d = str(decision["decision"]).strip().lower()
            if d in {"allow", "pass", "ok"}:
                return True, decision.get("return_value")
            if d in {"deny", "block"}:
                return False, decision.get("return_value")

    # Unknown format -> fail-open (do not break benchmark)
    return True, None


def _wrap_executor_method_with_gate(
    executor: Any, method_name: str, gate: Callable[..., Any],
    wrap_marker: str = "_ktc_d3_wrapped",
) -> bool:
    """
    Wraps a method on ToolsExecutor to enforce a pre-call gate.
    This is intentionally generic / best-effort.
    wrap_marker allows different defenses (D3, D4) to wrap independently.
    """
    if not hasattr(executor, method_name):
        return False

    original = getattr(executor, method_name)
    if not callable(original):
        return False

    # avoid double wrapping with the SAME marker
    if getattr(original, wrap_marker, False):
        return True

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            decision = gate(*args, **kwargs)
            allow, replacement = _interpret_gate_decision(decision)
            if not allow:
                # Das Gate hat tool_calls bereits in-place modifiziert (None bei
                # Full-Deny, gekürzte Liste bei Partial-Deny).  Statt das 5-Tupel
                # direkt zurückzugeben (Early-Return), wird der originale Executor
                # mit den modifizierten Messages aufgerufen.  Das ist notwendig,
                # weil ein Early-Return die ToolsExecutionLoop nicht terminiert —
                # die Loop erkennt nur über den normalen Executor-Rückgabepfad,
                # dass keine Tool-Calls mehr ausstehen.
                pass  # fall through to original(*args, **kwargs) below
        except Exception as e:
            # Fail-open to keep benchmark running, but print warning for debugging.
            _stderr(f"[custom-runner][WARN] gate error in {method_name}: {e}")
            _stderr(traceback.format_exc())

        return original(*args, **kwargs)

    setattr(_wrapped, wrap_marker, True)
    # Preserve other markers (so D3+D4 can coexist)
    for attr in ("_ktc_d3_wrapped", "_ktc_d4_wrapped"):
        if attr != wrap_marker and getattr(original, attr, False):
            setattr(_wrapped, attr, True)

    try:
        setattr(executor, method_name, _wrapped)
        return True
    except Exception:
        return False


def _wrap_pipeline_with_output_gate(pipeline: Any, disclosure_gate: Any) -> bool:
    """
    Wraps the pipeline's query method to apply the D3 output gate
    (disclosure guard) on the final assistant message before it's
    returned to the benchmark evaluator.

    Tries common pipeline method names: query, execute, run, __call__.
    """
    candidate_methods = ("query", "execute", "run", "__call__")

    for method_name in candidate_methods:
        if not hasattr(pipeline, method_name):
            continue
        original = getattr(pipeline, method_name)
        if not callable(original):
            continue
        if getattr(original, "_ktc_d3og_wrapped", False):
            return True  # already wrapped

        def _make_wrapper(orig_fn: Any) -> Any:
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                result = orig_fn(*args, **kwargs)

                # Apply output gate to the result's messages
                try:
                    _apply_output_gate_to_result(result, disclosure_gate)
                except Exception as e:
                    _stderr(f"[custom-runner][WARN] D3-OG filter error: {e}")

                return result
            setattr(_wrapped, "_ktc_d3og_wrapped", True)
            return _wrapped

        try:
            setattr(pipeline, method_name, _make_wrapper(original))
            _stdout(f"[custom-runner] D3-OG wrapped pipeline.{method_name}")
            return True
        except Exception:
            continue

    return False


def _apply_output_gate_to_result(result: Any, disclosure_gate: Any) -> None:
    """
    Best-effort: find messages in the pipeline result and apply the
    disclosure gate to the last assistant message.

    AgentDojo pipeline results can be:
    - A tuple (env, messages, ...)
    - An object with .messages attribute
    - A list of messages directly
    """
    messages = None

    # Case 1: result is a tuple/list of (something, messages, ...)
    if isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, list) and item and isinstance(item[0], dict) and "role" in item[0]:
                messages = item
                break

    # Case 2: result has .messages attribute
    if messages is None and hasattr(result, "messages"):
        m = getattr(result, "messages", None)
        if isinstance(m, list):
            messages = m

    # Case 3: result IS a list of messages
    if messages is None and isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and "role" in first:
            messages = result

    if messages is None:
        return

    # Apply disclosure gate
    disclosure_gate.filter_messages(messages)


def _attach_best_effort_hooks_to_pipeline(pipeline: Any, sel: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tries to inject D1/D2/D3/D4/D3-OG into the runtime pipeline instance.
    This is best-effort and intentionally defensive (won't crash on failure).
    """
    info: Dict[str, Any] = {
        "tools_executors_found": 0,
        "formatter_attached": False,
        "d3_gate_attached": False,
        "d3_gate_wrapped_method": None,
        "d4_gate_attached": False,
        "d4_gate_wrapped_method": None,
        "d3_output_gate_attached": False,
        "notes": [],
    }

    # If factory already knows how to patch the pipeline, prefer that
    apply_fn = _extract_apply_to_pipeline(sel)
    if apply_fn is not None:
        try:
            result = apply_fn(pipeline)
            info["notes"].append("apply_to_pipeline used from custom factory")
            info["apply_to_pipeline_result"] = _json_safe(result)
            return info
        except Exception as e:
            info["notes"].append(f"apply_to_pipeline failed, falling back to best-effort patching: {e}")

    formatter = _extract_formatter(sel)
    gate = _extract_call_gate(sel)

    def _make_event_logger() -> Callable[[Dict[str, Any]], None]:
        def _logger(event: Dict[str, Any]) -> None:
            if _ACTIVE_LOGDIR is None:
                return
            _append_jsonl(_ACTIVE_LOGDIR / "d3_decisions.jsonl", event)
        return _logger

    event_logger = _make_event_logger()

    if gate is not None and callable(getattr(gate, "set_event_logger", None)):
        try:
            gate.set_event_logger(event_logger)
        except Exception:
            info["notes"].append("Failed to attach event logger to D3 call gate")

    disclosure_gate = sel.get("disclosure_gate")
    if disclosure_gate is not None and callable(getattr(disclosure_gate, "set_event_logger", None)):
        try:
            disclosure_gate.set_event_logger(event_logger)
        except Exception:
            info["notes"].append("Failed to attach event logger to D3 output gate")

    # Extract D4 gate
    d4_gate = sel.get("d4_gate")
    if d4_gate is not None and not callable(d4_gate):
        d4_gate = None

    executors = _find_by_class_name(pipeline, "ToolsExecutor", max_depth=8)
    info["tools_executors_found"] = len(executors)

    # D1/D2: formatter on ToolsExecutor
    if formatter is not None:
        attached_any = False
        for ex in executors:
            if _set_tool_output_formatter_on_executor(ex, formatter):
                attached_any = True
        info["formatter_attached"] = attached_any
        if not attached_any:
            info["notes"].append("No compatible formatter attribute found on ToolsExecutor")
    else:
        info["notes"].append("No formatter found in defense selection (D1/D2 may be inactive)")

    # D3: pre-call gate (robust wie D4: alle relevanten Methoden wrappen)
    if gate is not None:
        wrapped = False
        wrapped_methods: List[str] = []
        candidate_methods = (
            "query",
            "execute_tool_calls",
            "execute_tool_call",
            "execute",
            "__call__",
            "run",
        )
        for ex in executors:
            for m in candidate_methods:
                if _wrap_executor_method_with_gate(ex, m, gate, wrap_marker="_ktc_d3_wrapped"):
                    wrapped = True
                    if m not in wrapped_methods:
                        wrapped_methods.append(m)

        info["d3_gate_attached"] = wrapped
        info["d3_gate_wrapped_method"] = ",".join(wrapped_methods) if wrapped_methods else None
        if not wrapped:
            info["notes"].append("No suitable ToolsExecutor method found for D3 gate wrapping")

    # D4: Least-Privilege gate (wraps ToolsExecutor like D3, but with own marker)
    if d4_gate is not None:
        wrapped = False
        wrapped_methods: List[str] = []
        candidate_methods = (
            "query",
            "execute_tool_calls",
            "execute_tool_call",
            "execute",
            "__call__",
            "run",
        )
        # IMPORTANT: For VLLM_PARSED / parsed tool calling, the executor method that
        # actually runs tools may NOT be query(). Therefore we wrap *all* commonly
        # used methods rather than stopping at the first one.
        for ex in executors:
            for m in candidate_methods:
                if _wrap_executor_method_with_gate(ex, m, d4_gate, wrap_marker="_ktc_d4_wrapped"):
                    wrapped = True
                    if m not in wrapped_methods:
                        wrapped_methods.append(m)

        info["d4_gate_attached"] = wrapped
        info["d4_gate_wrapped_method"] = ",".join(wrapped_methods) if wrapped_methods else None
        if not wrapped:
            info["notes"].append("No suitable ToolsExecutor method found for D4 gate wrapping")
    else:
        info["notes"].append("No D4 gate found in defense selection (D4 may be inactive)")

    # D3-OG: Output-Gate (disclosure guard) — wraps pipeline.query() to filter final output
    if disclosure_gate is not None and callable(getattr(disclosure_gate, "filter_messages", None)):
        attached = _wrap_pipeline_with_output_gate(pipeline, disclosure_gate)
        info["d3_output_gate_attached"] = attached
        if not attached:
            info["notes"].append("Failed to wrap pipeline with D3 output gate")
    else:
        info["notes"].append("No D3 output gate found in defense selection")

    return info


# ============================================================
# Monkeypatch AgentPipeline.from_config
# ============================================================

def _install_agentpipeline_patch(selection: Dict[str, Any], defense_ids: List[str], logdir: Path) -> None:
    global _PATCH_INSTALLED, _ORIGINAL_FROM_CONFIG_FUNC, _ACTIVE_SELECTION, _ACTIVE_DEFENSE_IDS, _ACTIVE_LOGDIR

    _ACTIVE_SELECTION = selection
    _ACTIVE_DEFENSE_IDS = list(defense_ids)
    _ACTIVE_LOGDIR = logdir

    if _PATCH_INSTALLED:
        return

    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline  # import here intentionally

    # Capture unbound function (classmethod -> function)
    current = AgentPipeline.from_config
    original_func = getattr(current, "__func__", current)
    _ORIGINAL_FROM_CONFIG_FUNC = original_func

    @classmethod
    def _patched_from_config(cls, config):  # type: ignore[override]
        pipeline = _ORIGINAL_FROM_CONFIG_FUNC(cls, config)

        try:
            attach_info = _attach_best_effort_hooks_to_pipeline(pipeline, _ACTIVE_SELECTION)
            # store lightweight metadata on pipeline for debugging if possible
            try:
                setattr(
                    pipeline,
                    "_ktc_custom_defense_runtime",
                    {
                        "defense_ids": list(_ACTIVE_DEFENSE_IDS),
                        "attach_info": attach_info,
                    },
                )
            except Exception:
                pass

            _stdout(
                "[custom-runner] Applied custom defenses to pipeline: "
                + json.dumps(_json_safe(attach_info), ensure_ascii=False)
            )
        except Exception as e:
            _stderr(f"[custom-runner][WARN] Failed to attach defenses to pipeline: {e}")
            _stderr(traceback.format_exc())

        return pipeline

    AgentPipeline.from_config = _patched_from_config  # type: ignore[assignment]
    _PATCH_INSTALLED = True


# ============================================================
# Forwarding to official AgentDojo benchmark CLI
# ============================================================

def _invoke_agentdojo_benchmark_cli(forward_args: List[str]) -> int:
    """
    Call the official click-based AgentDojo benchmark command so wrappers like
    VLLM_PARSED are handled exactly like in `python -m agentdojo.scripts.benchmark`.
    """
    from agentdojo.scripts import benchmark as ad_benchmark

    # ad_benchmark.main is a click command object (decorator @click.command)
    click_cmd = ad_benchmark.main

    try:
        # Preferred for click commands
        if hasattr(click_cmd, "main"):
            ret = click_cmd.main(args=forward_args, prog_name="agentdojo-benchmark", standalone_mode=False)
        else:
            # Fallback (rare)
            old_argv = sys.argv[:]
            try:
                sys.argv = ["agentdojo-benchmark", *forward_args]
                ret = click_cmd()
            finally:
                sys.argv = old_argv
    except SystemExit as e:
        # Click or benchmark code may call exit(...)
        code = e.code if isinstance(e.code, int) else 1
        return code
    except Exception:
        raise

    if isinstance(ret, int):
        return ret
    return 0


# ============================================================
# CLI parsing
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Custom AgentDojo benchmark runner with thesis defense selection and pipeline patching."
    )

    # Common args passed by your orchestrator
    p.add_argument("--model", required=True, help="AgentDojo model wrapper / model arg (e.g., VLLM_PARSED)")
    p.add_argument("--model-id", required=True, help="Model identifier")
    p.add_argument("--logdir", required=True, help="Benchmark output directory")
    p.add_argument("--attack", default=None, help="Attack name, e.g. tool_knowledge")
    p.add_argument("--benchmark-version", default=None, help="Benchmark version, e.g. v1.2.2")

    # Optional commonly-used passthrough args (known here so they appear in dry-run JSON)
    p.add_argument("--tool-delimiter", default=None)
    p.add_argument("--system-message-name", default=None)
    p.add_argument("--system-message", default=None)
    p.add_argument("--tool-output-format", default=None)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--force-rerun", action="store_true")

    # Custom args for thesis defenses
    p.add_argument(
        "--defense-id",
        action="append",
        default=[],
        help=(
            "Repeatable defense ID(s), e.g. d1_delimiting, d1_datamarking, d1_encoding, "
            "d2_json, d3_balanced, d3_strict, d3_output_gate"
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Resolve defenses and print config without running benchmark")

    return p


def _compose_forward_args(args: argparse.Namespace, unknown_forward_args: List[str]) -> List[str]:
    """
    Build argv for agentdojo.scripts.benchmark (official CLI), excluding custom-only flags.
    """
    out: List[str] = []

    def add_opt(name: str, value: Any) -> None:
        if value is None:
            return
        out.extend([name, str(value)])

    def add_flag(name: str, enabled: bool) -> None:
        if enabled:
            out.append(name)

    add_opt("--model", args.model)
    add_opt("--model-id", args.model_id)
    add_opt("--logdir", args.logdir)
    add_opt("--attack", args.attack)
    add_opt("--benchmark-version", args.benchmark_version)
    add_opt("--tool-delimiter", args.tool_delimiter)
    add_opt("--system-message-name", args.system_message_name)
    add_opt("--system-message", args.system_message)
    add_opt("--tool-output-format", args.tool_output_format)
    add_opt("--max-workers", args.max_workers)
    add_flag("--force-rerun", bool(args.force_rerun))

    # Ensure deterministic patching (monkeypatch + multiprocessing can get messy)
    if args.max_workers is None and not any(tok == "--max-workers" for tok in unknown_forward_args):
        out.extend(["--max-workers", "1"])

    # Important: do NOT forward --defense-id / --dry-run (custom-only)
    # But do forward anything else the user/orchestrator adds (e.g. -s, -ut, -ml, ...)
    out.extend(unknown_forward_args)

    return out


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = _build_parser()
    args, unknown_forward_args = parser.parse_known_args()

    logdir = Path(args.logdir)
    _ensure_dir(logdir)

    # Resolve defense selection
    build_defense_selection = _load_build_defense_selection()
    try:
        selection_raw = build_defense_selection(list(args.defense_id))
    except Exception as e:
        _stderr(f"[custom-runner][ERROR] Failed to resolve defense selection: {e}")
        return 2
    selection = _normalize_selection(selection_raw)

    # Write a trace artifact for reproducibility
    selection_artifact = {
        "defense_ids": list(args.defense_id),
        "model_wrapper": args.model,
        "model_id": args.model_id,
        "attack": args.attack,
        "benchmark_version": args.benchmark_version,
        "logdir": str(logdir),
        "factory_metadata": selection.get("metadata", {}),
        "selection_keys": sorted(list(selection.keys())),
        "unknown_forward_args": list(unknown_forward_args),
        "notes": (
            "Custom runner patches AgentPipeline.from_config and forwards execution "
            "to agentdojo.scripts.benchmark CLI flow."
        ),
    }
    _write_json(logdir / "custom_defense_selection.json", selection_artifact)
    _stdout("[custom-runner] Defense selection resolved and written to custom_defense_selection.json")

    if args.dry_run:
        preview = {
            "defense_ids": list(args.defense_id),
            "model_wrapper": args.model,
            "model_id": args.model_id,
            "attack": args.attack,
            "benchmark_version": args.benchmark_version,
            "logdir": str(logdir),
            "factory_metadata": selection.get("metadata", {}),
            "unknown_forward_args": list(unknown_forward_args),
            "forward_args": _compose_forward_args(args, unknown_forward_args),
            "notes": "Dry run only; no benchmark executed.",
        }
        print(json.dumps(_json_safe(preview), indent=2, ensure_ascii=False))
        return 0

    # Install pipeline patch before benchmark starts
    _install_agentpipeline_patch(selection, list(args.defense_id), logdir)
    
    # Forward to official AgentDojo benchmark CLI flow (this fixes VLLM_PARSED enum issue)
    forward_args = _compose_forward_args(args, unknown_forward_args)

    try:
        return _invoke_agentdojo_benchmark_cli(forward_args)
    except Exception as e:
        _stderr(f"[custom-runner][ERROR] Benchmark execution failed: {e}")
        _stderr(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())