"""
D3 Output-Gate: Deterministischer Filter für Modellantworten
=============================================================
Enforcement auf dem Response-Pfad (Agent -> Evaluator).
Filtert die finale Textantwort des Modells VOR der Rückgabe.

Regelkategorien (vgl. Abschnitt 5.5.4):
  1. Injektionsblock-Entfernung (<INFORMATION>-Tags)
  2. Redaktion unbekannter IBANs (partielle Allowlist)
  3. Maskierung offensichtlicher Tool-Call-Artefakte
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Set


REDACTION_MARKER = "[REDACTED]"

_INFO_BLOCK_CLOSED = re.compile(
    r"<INFORMATION>.*?</INFORMATION>",
    re.DOTALL | re.IGNORECASE,
)
_INFO_BLOCK_UNCLOSED = re.compile(
    r"<INFORMATION>.*$",
    re.DOTALL | re.IGNORECASE,
)
_IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b")
_TOOL_ARTIFACT_PATTERN = re.compile(
    r"(?:\[TOOL_CALLS\]|tool_calls|\"tool_calls\"\s*:|\"function\"\s*:|function\s*=)",
    re.IGNORECASE,
)

# Bekannte legitime IBANs (AgentDojo-/Task-Kontext)
IBAN_ALLOWLIST: Set[str] = {
    "CH9300762011623852957",       # Emma Johnson (AgentDojo)
    "GB29NWBK60161331926819",      # Bekannte legitime Gegenpartei
}



def filter_output(text: str, task_prompt: str = "") -> str:
    """
    Filtert finale Modellantworten vor der Rückgabe an den Evaluator/Nutzer.

    Args:
        text:         Zu filternder Text
        task_prompt:  AgentDojo-Task-Prompt (für dynamische IBAN-Allowlist)

    Returns:
        Gefilterter Text mit REDACTION_MARKER wo nötig.
    """
    if not text:
        return text

    result = text

    # ── Regel 1: Injektionsblock-Entfernung ─────────────────────
    result = _INFO_BLOCK_CLOSED.sub(REDACTION_MARKER, result)
    result = _INFO_BLOCK_UNCLOSED.sub(REDACTION_MARKER, result)

    # ── Regel 2: Unbekannte IBANs redaktieren ───────────────────
    prompt_ibans = set(_IBAN_PATTERN.findall(task_prompt or ""))
    allowed = {x.upper() for x in IBAN_ALLOWLIST | prompt_ibans}

    def _replace_iban(match: re.Match[str]) -> str:
        iban = match.group(0)
        return iban if iban.upper() in allowed else REDACTION_MARKER

    result = _IBAN_PATTERN.sub(_replace_iban, result)

    # ── Regel 3: Offensichtliche Tool-Call-Artefakte ────────────
    result = _TOOL_ARTIFACT_PATTERN.sub(REDACTION_MARKER, result)

    return result


class DisclosureGate:
    """
    Finales Output-Gate auf dem Response-Pfad.
    Erwartet eine Liste von Nachrichten und filtert die letzte Assistant-Nachricht.

    Wichtig: Content-Blöcke werden konsistent als
    {"type": "text", "content": ...} normalisiert.
    """

    def __init__(
        self,
        task_prompt_getter: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
        event_logger: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._task_prompt_getter = task_prompt_getter
        self._event_logger = event_logger

    def set_event_logger(self, event_logger: Callable[[Dict[str, Any]], None]) -> None:
        self._event_logger = event_logger

    def _extract_text(self, msg: Dict[str, Any]) -> str:
        content = msg.get("content")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("content")
                    if not isinstance(text, str):
                        text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)

        return ""

    def _replace_text(self, msg: Dict[str, Any], new_text: str) -> None:
        msg["content"] = [{"type": "text", "content": new_text}]

    def filter_messages(self, messages: List[Dict[str, Any]]) -> None:
        if not isinstance(messages, list) or not messages:
            return

        task_prompt = ""
        if callable(self._task_prompt_getter):
            try:
                task_prompt = self._task_prompt_getter(messages) or ""
            except Exception:
                task_prompt = ""

        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue

            original = self._extract_text(msg)
            filtered = filter_output(original, task_prompt=task_prompt)

            if filtered != original:
                self._replace_text(msg, filtered)
                if callable(self._event_logger):
                    self._event_logger(
                        {
                            "gate_type": "output_gate",
                            "decision": "transform",
                            "reason_code": "OUTPUT_FILTER_APPLIED",
                            "risk_score": 60,
                            "original_length": len(original),
                            "filtered_length": len(filtered),
                        }
                    )
            return
