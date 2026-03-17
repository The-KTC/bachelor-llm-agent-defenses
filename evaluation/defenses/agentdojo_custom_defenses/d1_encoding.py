from __future__ import annotations
from typing import Any, Optional
from .common import b64_utf8, default_tool_name, dumps_pretty_json, safe_to_python

def encoding_formatter(
    tool_result: Any,
    *,
    tool_name: Optional[str] = None,
    encoding: str = "base64_json_utf8",
) -> str:

    if encoding != "base64_json_utf8":
        raise ValueError(f"Unsupported encoding: {encoding}")

    payload_json = dumps_pretty_json(safe_to_python(tool_result))
    payload_b64 = b64_utf8(payload_json)
    return (
        "UNTRUSTED TOOL OUTPUT (ENCODED)\n"
        f"tool={default_tool_name(tool_name)}\n"
        f"encoding={encoding}\n"
        "payload_b64=\n"
        f"{payload_b64}"
    )

