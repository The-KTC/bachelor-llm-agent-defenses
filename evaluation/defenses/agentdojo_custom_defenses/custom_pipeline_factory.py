"""
Custom Pipeline Factory für AgentDojo Bachelor-Thesis Defenses
===============================================================
Baut aus einer Liste von Defense-IDs die entsprechenden Komponenten
(Formatter, Call-Gate, Output-Gate) und gibt sie als DefenseSelection
zurück, die vom run_benchmark_custom_pipeline.py in die AgentDojo-
Pipeline eingehängt wird.

Unterstützte Defense-IDs:
  - d1_delimiting, d1_datamarking, d1_encoding
  - d2_json
  - d3_balanced, d3_strict
  - d3_output_gate
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .d1_delimiting import delimiting_formatter
from .d1_datamarking import datamarking_formatter
from .d1_encoding import encoding_formatter
from .d2_structured_output_json import structured_json_formatter
from .d3_call_gate import CallGate
from .d3_output_gate import DisclosureGate

Formatter = Callable[[Any], str]

SUPPORTED_DEFENSE_IDS = {
    "d1_delimiting",
    "d1_datamarking",
    "d1_encoding",
    "d2_json",
    "d3_balanced",
    "d3_strict",
    "d3_output_gate",
}


@dataclass
class DefenseSelection:
    defense_ids: List[str]
    formatter: Optional[Formatter] = None
    call_gate: Optional[Any] = None
    disclosure_gate: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _chain_formatters(formatters: Sequence[Formatter]) -> Formatter:
    """Verkettet mehrere Formatter sequenziell."""
    def _fmt(tool_result: Any) -> str:
        value: Any = tool_result
        for fn in formatters:
            value = fn(value)
        return value if isinstance(value, str) else str(value)
    return _fmt


def _extract_task_prompt_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Extrahiert die erste User-Nachricht aus der Message-History."""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue

        content = msg.get("content")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    elif isinstance(block.get("content"), str):
                        parts.append(block["content"])
                elif isinstance(block, str):
                    parts.append(block)
            if parts:
                return "\n".join(parts)

        break
    return ""


def build_defense_selection(defense_ids: Sequence[str]) -> DefenseSelection:
    """
    Hauptfunktion: Baut aus Defense-IDs die Pipeline-Komponenten.
    Wird von run_benchmark_custom_pipeline.py aufgerufen.
    """
    ids = list(defense_ids or [])
    unknown = sorted(set(ids) - SUPPORTED_DEFENSE_IDS)
    if unknown:
        raise ValueError(
            f"Unsupported defense-id(s): {unknown}. Supported: {sorted(SUPPORTED_DEFENSE_IDS)}"
        )

    fmt_list: List[Formatter] = []
    call_gate = None
    disclosure_gate = None

    # ── D1: Spotlighting-Formatter ──────────────────────────────
    if "d1_delimiting" in ids:
        fmt_list.append(lambda x: delimiting_formatter(x))
    if "d1_datamarking" in ids:
        fmt_list.append(lambda x: datamarking_formatter(x))
    if "d1_encoding" in ids:
        fmt_list.append(lambda x: encoding_formatter(x))

    # ── D2: Structured Output (JSON-Envelope) ───────────────────
    if "d2_json" in ids:
        d2 = lambda x: structured_json_formatter(x)
        if fmt_list:
            fmt_list.insert(0, d2)  # D2 vor D1, da D1/D2 den Text weiter transformieren
        else:
            fmt_list.append(d2)

    # ── D3: Call-Gate ───────────────────────────────────────────
    if "d3_balanced" in ids:
        call_gate = CallGate(mode="balanced")
    elif "d3_strict" in ids:
        call_gate = CallGate(mode="strict")

    # ── Formatter für D1/D2 ─────────────────────────────────
    formatter = _chain_formatters(fmt_list) if fmt_list else None

    # ── D3: Output-Gate auf finalem Response-Pfad ───────────────
    if "d3_output_gate" in ids:
        disclosure_gate = DisclosureGate(task_prompt_getter=_extract_task_prompt_from_messages)

    # ── Metadata für Logging / Reproduzierbarkeit ───────────────
    meta = {
        "defense_ids": ids,
        "supported_defense_ids": sorted(SUPPORTED_DEFENSE_IDS),
        "has_formatter": formatter is not None,
        "has_call_gate": call_gate is not None,
        "call_gate_mode": getattr(call_gate, "mode", None),
        "has_disclosure_gate": disclosure_gate is not None,
        "has_output_gate": disclosure_gate is not None,
        "output_gate_applies_to": ["final_response"] if disclosure_gate is not None else [],
        "tool_output_formatter_active": formatter is not None,
    }

    return DefenseSelection(
        defense_ids=ids,
        formatter=formatter,
        call_gate=call_gate,
        disclosure_gate=disclosure_gate,
        metadata=meta,
    )
