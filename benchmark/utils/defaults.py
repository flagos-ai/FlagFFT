"""Shared benchmark pytest defaults."""

DEFAULTS = {
    "warmup": 10,
    "iters": 100,
    "csv": None,  # None = auto-generate path; "" = disabled
    "suite": "typical",
}
