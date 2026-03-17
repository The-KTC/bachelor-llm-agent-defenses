from __future__ import annotations
import json
from typing import Any, Optional
from .common import default_tool_name, safe_to_python

def structured_json_formatter(
    tool_result: Any,
    *,
    tool_name: Optional[str] = None,
    trusted: bool = False,
    schema_version: str = "1.0",
) -> str:
    payload = safe_to_python(tool_result)
    envelope = {
        "schema": "tool_result_envelope",
        "schema_version": schema_version,
        "trusted": trusted,
        "source_type": "tool",
        "tool_name": default_tool_name(tool_name),
        "payload": payload,
    }
    return json.dumps(envelope, ensure_ascii=False, indent=2)

