"""Legacy epoch-ai-cup package.

Keep package import side effects minimal so pilot tooling can import lightweight
modules (for example `src.data` or `src.validate`) without eagerly importing the
entire research stack.
"""

__all__ = [
    "data",
    "features",
    "metrics",
    "submission",
    "postprocessing",
    "validate",
]
