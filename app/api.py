"""
FastAPI service placeholder.

Endpoints to add later:
- POST /detect : accepts time series
"""

from __future__ import annotations

try:
    from fastapi import FastAPI
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore

app = FastAPI() if FastAPI is not None else None


def create_app():
    if app is None:
        raise RuntimeError(
            "FastAPI is not installed. Install `fastapi` and `uvicorn` to run the API."
        )
    return app

