from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from src.biomarker_engine import list_cancer_types, rank_candidates
from src.genai_synthesizer import synthesize_decision
from pipelines.ingest import download_csv, ingest_csv_to_parquet, curate_parquet_to_silver

app = FastAPI(
    title="Cancer Biomarker Platform API",
    description="FastAPI API for biomarker ranking, ingestion, and GenAI synthesis.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health_check() -> dict:
    return {
        "status": "healthy",
        "service": "cancer-biomarker-platform",
        "version": "0.2.0",
    }


@app.get("/api/v1/cancer-types")
def cancer_types() -> dict:
    return {"cancer_types": list_cancer_types()}


@app.get("/api/v1/candidates")
def get_candidates(
    cancer_type: str = Query(..., description="Cancer type, e.g. PDAC or BRCA"),
    limit: int = Query(5, ge=1, le=50),
):
    results = rank_candidates(cancer_type=cancer_type, limit=limit)
    return results.to_dict(orient="records")


class IngestResponse(BaseModel):
    bronze_path: str
    silver_path: str


@app.post("/api/v1/ingest")
def ingest_from_url(url: str):
    """Ingest a CSV from a URL into bronze and silver layers. Returns paths to created parquet files."""
    try:
        csv_path = download_csv(url)
        bronze = ingest_csv_to_parquet(csv_path)
        silver = curate_parquet_to_silver(bronze)
        return IngestResponse(bronze_path=str(bronze), silver_path=str(silver))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/synthesize")
def synthesize(cancer_type: str = Query(...), question: str | None = None):
    # Get top candidates
    candidates = rank_candidates(cancer_type=cancer_type, limit=10).to_dict(orient="records")
    synthesis = synthesize_decision(candidates, question=question)
    return {"candidates": candidates, "synthesis": synthesis}


@app.get("/")
def root() -> dict:
    return {
        "message": "Cancer Biomarker Platform API",
        "docs": "/docs",
    }
