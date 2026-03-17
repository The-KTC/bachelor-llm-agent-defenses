"""
D3 Call-Gate: Policy-basierte Prüfung von Tool-Calls
=====================================================
Enforcement auf dem Call-Pfad (Agent -> Tool).
Prüft jeden Tool-Call VOR der Ausführung gegen deterministische Regeln.

Integration: Wird als callable an den ToolsExecutor übergeben.
             Im Deny-/Partial-Deny-Fall wird die letzte Assistant-
             Nachricht normalisiert/ersetzt, damit die nachfolgende
             Serialisierung API-kompatibel bleibt.

Zwei Modi (vgl. Abschnitt 5.5.3):
  - balanced: Task-Prompt-Binding — schreibende Calls werden nur
              erlaubt, wenn ihre Ziel-Identifikatoren (IBANs, E-Mails,
              URLs) im Task-Prompt vorkommen.
  - strict:   Deny-by-default — nur lesende Tools erlaubt.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    TRANSFORM = "transform"


@dataclass
class PolicyDecision:
    """Ergebnis einer PDP-Prüfung, inkl. reason_code für Logging (R6)."""
    decision: Decision
    reason_code: str
    message: str = ""
    risk_score: int = 0


READ_ONLY_PREFIXES = (
    "get_", "read_", "list_", "search_",
    "lookup_", "fetch_", "retrieve_",
)

ACTION_PATTERN = re.compile(
    r"(?:^|_)(send|post|publish|upload|share|invite|add|remove|"
    r"delete|create|update|transfer|pay|book|reserve|cancel|append)(?:_|$)",
    re.IGNORECASE,
)

_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_URL_RE = re.compile(r"https?://[^\s\"',\]}>]+", re.IGNORECASE)


def _iter_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)



def _extract_identifiers(text: str) -> Set[str]:
    ids: Set[str] = set()
    ids.update(_IBAN_RE.findall(text or ""))
    ids.update(m.lower() for m in _EMAIL_RE.findall(text or ""))
    ids.update(m.lower() for m in _URL_RE.findall(text or ""))
    return ids



def _extract_task_prompt(messages: list) -> str:
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role != "user":
            continue
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    if isinstance(block.get("text"), str):
                        parts.append(block["text"])
                    elif isinstance(block.get("content"), str):
                        parts.append(block["content"])
                elif isinstance(block, str):
                    parts.append(block)
            if parts:
                return "\n".join(parts)
        break
    return ""



def _extract_name_args(tc: Any) -> tuple[str, Dict[str, Any]]:
    fn = getattr(tc, "function", None)
    argd = getattr(tc, "args", None)
    if isinstance(fn, str) and isinstance(argd, dict):
        return fn, argd

    if isinstance(tc, dict):
        fn_obj = tc.get("function")
        if isinstance(fn_obj, dict):
            name = fn_obj.get("name", "unknown_tool")
            arguments = fn_obj.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {"_raw": arguments}
            return str(name), dict(arguments) if isinstance(arguments, dict) else {}

        name = tc.get("name") or tc.get("function") or "unknown_tool"
        arguments = tc.get("arguments") or tc.get("args") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {"_raw": arguments}
        return str(name), dict(arguments) if isinstance(arguments, dict) else {}

    return "unknown_tool", {}



def decide_call(
    tool_name: str,
    arguments: Dict[str, Any],
    mode: str = "balanced",
    task_identifiers: Optional[Set[str]] = None,
) -> PolicyDecision:
    name_lower = tool_name.lower()

    if mode == "strict":
        if name_lower.startswith(READ_ONLY_PREFIXES):
            return PolicyDecision(Decision.ALLOW, "STRICT_READONLY", risk_score=5)
        return PolicyDecision(
            Decision.DENY, "STRICT_POLICY",
            message=f"Tool '{tool_name}' ist im strict-Modus nicht erlaubt",
            risk_score=80,
        )

    if name_lower.startswith(READ_ONLY_PREFIXES):
        return PolicyDecision(Decision.ALLOW, "BALANCED_READONLY", risk_score=5)

    if ACTION_PATTERN.search(name_lower) and task_identifiers is not None:
        arg_ids: Set[str] = set()
        for s in _iter_strings(arguments):
            arg_ids.update(_IBAN_RE.findall(s))
            arg_ids.update(m.lower() for m in _EMAIL_RE.findall(s))
            arg_ids.update(m.lower() for m in _URL_RE.findall(s))

        unknown = arg_ids - task_identifiers
        if unknown:
            return PolicyDecision(
                Decision.DENY, "TASK_BINDING_VIOLATION",
                message=f"Ziel nicht im Task-Prompt: {sorted(unknown)}",
                risk_score=90,
            )

    return PolicyDecision(Decision.ALLOW, "ALLOW_DEFAULT", risk_score=5)


# ── Hilfsfunktionen für robuste Message-Normalisierung ───────────────

def _normalize_content_blocks(content: Any) -> list[dict[str, Any]] | None:
    if content is None:
        return None

    if isinstance(content, str):
        return [{"type": "text", "content": content}]

    if isinstance(content, list):
        out: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, str):
                out.append({"type": "text", "content": block})
            elif isinstance(block, dict):
                text = block.get("content")
                if not isinstance(text, str):
                    text = block.get("text")
                if text is None:
                    text = ""
                out.append({"type": "text", "content": str(text)})
            else:
                out.append({"type": "text", "content": str(block)})
        return out or None

    return [{"type": "text", "content": str(content)}]



def _message_role(msg: Any) -> str:
    if isinstance(msg, dict):
        role = msg.get("role")
    else:
        role = getattr(msg, "role", None)
    return role if isinstance(role, str) and role else "assistant"



def _message_content(msg: Any) -> Any:
    if isinstance(msg, dict):
        return msg.get("content")
    return getattr(msg, "content", None)



def _replace_last_assistant_message(
    messages: list[Any],
    *,
    role: str = "assistant",
    content: Any = None,
    tool_calls: Any = None,
) -> None:
    normalized_tool_calls = tool_calls if isinstance(tool_calls, list) and tool_calls else None
    messages[-1] = {
        "role": role,
        "content": _normalize_content_blocks(content),
        "tool_calls": normalized_tool_calls,
    }


class CallGate:
    """
    Callable Gate für AgentDojo-Pipeline-Integration.
    Normalisiert/ersetzt messages[-1] im Deny-/Partial-Deny-Fall,
    statt gefrorene Message-Objekte in-place zu patchen.
    """

    def __init__(
        self,
        mode: str = "balanced",
        event_logger: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.mode = mode
        self._debug = os.getenv("D3_DEBUG", "0") == "1"
        self._event_logger = event_logger

    def set_event_logger(self, event_logger: Callable[[Dict[str, Any]], None]) -> None:
        self._event_logger = event_logger

    def _emit_event(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        decision: PolicyDecision,
        task_ids: Set[str],
    ) -> None:
        if callable(self._event_logger):
            self._event_logger(
                {
                    "gate_type": "call_gate",
                    "mode": self.mode,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "decision": decision.decision.value,
                    "reason_code": decision.reason_code,
                    "risk_score": decision.risk_score,
                    "message": decision.message,
                    "task_identifiers": sorted(task_ids),
                }
            )

    def __call__(self, *args, **kwargs):
        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            for a in args:
                if isinstance(a, list) and a:
                    first = a[0]
                    if (isinstance(first, dict) and "role" in first) or hasattr(first, "role"):
                        messages = a
                        break
        if not messages:
            return None

        task_prompt = _extract_task_prompt(messages)
        task_ids = _extract_identifiers(task_prompt) if task_prompt else set()

        last = messages[-1]
        tool_calls = (
            last.get("tool_calls") if isinstance(last, dict)
            else getattr(last, "tool_calls", None)
        )
        if not isinstance(tool_calls, list) or not tool_calls:
            return None

        kept = []
        denied: List[PolicyDecision] = []

        for tc in tool_calls:
            name, argd = _extract_name_args(tc)
            decision = decide_call(name, argd, mode=self.mode, task_identifiers=task_ids)
            self._emit_event(name, argd, decision, task_ids)

            if decision.decision == Decision.ALLOW:
                kept.append(tc)
                if self._debug:
                    print(f"[D3-CG] ALLOW {name}: {decision.reason_code}", flush=True)
            else:
                denied.append(decision)
                if self._debug:
                    print(f"[D3-CG] DENY  {name}: {decision.reason_code} - {decision.message}", flush=True)

        original_content = _message_content(last)
        original_role = _message_role(last)

        # Full deny: keine Tool-Calls mehr -> saubere Assistant-Nachricht mit Textinhalt.
        if denied and not kept:
            first = denied[0]
            denial_msg = (
                f"[POLICY_DENIED] reason_code={first.reason_code}; "
                f"risk_score={first.risk_score}; message={first.message}"
            )

            _replace_last_assistant_message(
                messages,
                role=original_role,
                content=denial_msg,
                tool_calls=None,
            )

            return {
                "decision": "deny",
                "allow": False,
                "return_value": denial_msg,
            }

        # Partial deny / Filterung: letzte Nachricht ebenfalls normalisieren,
        # aber erlaubte Tool-Calls beibehalten.
        if denied:
            _replace_last_assistant_message(
                messages,
                role=original_role,
                content=original_content,
                tool_calls=kept,
            )

        return None
