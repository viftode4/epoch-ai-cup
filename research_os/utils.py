from __future__ import annotations

import re

import numpy as np


_slug_pattern = re.compile(r"[^a-z0-9]+")


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def slugify(value: str) -> str:
    text = _slug_pattern.sub("-", value.strip().lower()).strip("-")
    return text or "workflow"
