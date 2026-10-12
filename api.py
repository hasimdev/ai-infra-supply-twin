"""FastAPI service + dashboard.

  uvicorn api:app --reload      # then open http://localhost:8000
"""
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from engine import get_xp
from pipeline import analyze

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="AI Infra Supply Twin",
              description="GPU Monte Carlo + LLM disruption feed for the AI hardware supply chain.")


class AnalyzeRequest(BaseModel):
    runs: int = Field(100_000, ge=1_000, le=5_000_000)
    backend: Literal["auto", "cupy", "numpy"] = "auto"
    live: bool = False
    min_confidence: float = Field(0.6, ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok", "backend": get_xp("auto")[1]}


@app.post("/analyze")
def run_analysis(req: AnalyzeRequest):
    try:
        return analyze(req.runs, req.backend, req.live, req.min_confidence)
    except ImportError as e:
        raise HTTPException(400, f"Backend unavailable: {e}")
    except RuntimeError as e:
        raise HTTPException(502, str(e))


@app.get("/")
def dashboard():
    return FileResponse(STATIC / "index.html")
