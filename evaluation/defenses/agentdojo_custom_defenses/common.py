from __future__ import annotations

import base64
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Optional


def safe_to_python(obj: Any) -> Any:
    """Best-effort conversion of arbitrary objects into JSON-serializable data."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): safe_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [safe_to_python(x) for x in obj]
    if is_dataclass(obj):
        return safe_to_python(asdict(obj))
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return safe_to_python(fn())
            except Exception:
                pass
    try:
        return str(obj)
    except Exception:
        return f"<unserializable:{type(obj).__name__}>"


def dumps_pretty_json(data: Any) -> str:
    return json.dumps(safe_to_python(data), ensure_ascii=False, indent=2, sort_keys=True)


def try_yaml_dump(data: Any) -> str:
    """Try YAML; fall back to JSON if PyYAML is unavailable."""
    try:
        import yaml  # type: ignore
        return yaml.safe_dump(
            safe_to_python(data),
            sort_keys=True,
            allow_unicode=True,
            default_flow_style=False,
        )
    except Exception:
        return dumps_pretty_json(data)


def b64_utf8(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def default_tool_name(tool_name: Optional[str]) -> str:
    return tool_name or "unknown_tool"
