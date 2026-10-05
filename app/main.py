from fastapi import FastAPI, Query, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import io
import pandas as pd
from datetime import datetime

from src.biomarker_engine import get_sample_data, calculate_biomarker_scores, list_cancer_types
from src.genai_synthesizer import synthesize_decision_with_genai

app = FastAPI(
    title="Cancer Biomarker Platform API",
    description="AI-powered biomarker ranking and decision-support platform for oncology.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SynthesisRequest(BaseModel):
    cancer_type: str
    use_case: str = "diagnostic"
    question: Optional[str] = None
    top_n: int = 5


class IngestUploadResponse(BaseModel):
    status: str
    message: str
    genes_uploaded: int
    genes_processed: int
    cancer_type: str
    use_case: str
    timestamp: str
    top_candidates: list


@app.get("/api/v1/health")
def health_check() -> dict:
    return {
        "status": "healthy",
        "service": "cancer-biomarker-platform",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/info")
def info() -> dict:
    return {
        "name": "Cancer Biomarker Platform",
        "description": "Biomarker prioritization and AI decision support for oncology workflows.",
        "use_cases": ["diagnostic", "prognostic", "drug_response", "treatment_monitoring"],
        "supported_cancer_types": [
            "PDAC", "BRCA", "LUAD", "LIHC", "OV", "PRAD", "COAD",
        ],
    }


@app.get("/api/v1/cancer-types")
def cancer_types() -> dict:
    return {"cancer_types": list_cancer_types()}


@app.get("/api/v1/candidates")
def get_candidates(
    cancer_type: str = Query(..., description="Cancer type to score, e.g. PDAC or BRCA"),
    use_case: str = Query("diagnostic", description="diagnostic | prognostic | drug_response | treatment_monitoring"),
    limit: int = Query(5, ge=1, le=20),
):
    data = get_sample_data()
    results = calculate_biomarker_scores(data=data, cancer_type=cancer_type, use_case=use_case, limit=limit)
    return {
        "cancer_type": cancer_type,
        "use_case": use_case,
        "count": len(results),
        "candidates": results.to_dict(orient="records"),
    }


@app.get("/api/v1/candidate-detail")
def candidate_detail(
    gene: str = Query(...),
    cancer_type: str = Query(...),
):
    data = get_sample_data()
    match = data[(data["gene_symbol"] == gene.upper()) & (data["cancer_type"] == cancer_type.upper())]
    if match.empty:
        raise HTTPException(status_code=404, detail="Candidate not found for this cancer type.")
    row = match.iloc[0]
    return {
        "gene": row["gene_symbol"],
        "name": row.get("gene_name", row["gene_symbol"]),
        "cancer_type": row["cancer_type"],
        "description": row.get("description", ""),
        "mechanism": row.get("biological_mechanism", ""),
        "clinical_relevance": row.get("clinical_relevance", ""),
        "tumor_specificity": float(row.get("specificity_score", 0.0) or 0.0),
        "off_target_risk": float(row.get("off_target_risk", 0.0) or 0.0),
        "surface_accessible": bool(row.get("surface_accessible", False)),
        "ev_association": float(row.get("tumor_normal_ratio", 0.0) or 0.0),
        "reagent_availability": float(row.get("evidence_strength", 0.0) or 0.0),
    }


@app.post("/api/v1/synthesize")
def synthesize(request: SynthesisRequest):
    data = get_sample_data()
    candidates_df = calculate_biomarker_scores(
        data=data,
        cancer_type=request.cancer_type,
        use_case=request.use_case,
        limit=request.top_n,
    )
    candidates = candidates_df.to_dict(orient="records")
    result = synthesize_decision_with_genai(
        cancer_type=request.cancer_type,
        use_case=request.use_case,
        candidates=candidates,
        question=request.question,
    )
    return {
        "cancer_type": request.cancer_type,
        "use_case": request.use_case,
        "candidates": candidates,
        "summary": result["summary"],
        "recommendation": result["recommendation"],
        "evidence_sources": result["evidence_sources"],
        "caveats": result["caveats"],
        "next_experiments": result["next_experiments"],
        "confidence_score": result["confidence_score"],
    }


@app.post("/api/v1/upload-data")
async def upload_data(file: UploadFile = File(...), cancer_type: str = Query(...), use_case: str = Query("diagnostic")):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        required = {"gene_symbol", "expression_value"}
        if not required.issubset(df.columns):
            raise ValueError("CSV must contain at least 'gene_symbol' and 'expression_value' columns.")

        df = df.copy()
        df["gene_symbol"] = df["gene_symbol"].astype(str).str.upper()
        df["cancer_type"] = cancer_type.upper()
        df["specificity_score"] = 0.7 + (df["expression_value"] / max(df["expression_value"].max(), 1.0)) * 0.3
        df["off_target_risk"] = 0.1
        df["response_signal"] = df["expression_value"] / max(df["expression_value"].max(), 1.0)
        df["evidence_strength"] = 0.75
        df["surface_accessible"] = True
        df["ev_association_score"] = df["response_signal"]
        df["reagent_availability"] = 0.9

        results = calculate_biomarker_scores(data=df, cancer_type=cancer_type, use_case=use_case, limit=5)
        return {
            "status": "success",
            "message": f"Processed {len(df)} genes for {cancer_type}.",
            "genes_uploaded": int(len(df)),
            "genes_processed": int(len(results)),
            "cancer_type": cancer_type,
            "use_case": use_case,
            "timestamp": datetime.now().isoformat(),
            "top_candidates": results.to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(exc)}")


@app.get("/api/v1/dual-marker-gates")
def dual_marker_gates(cancer_type: str = Query(...), top_n: int = Query(5, ge=2, le=10)):
    data = get_sample_data()
    ranked = calculate_biomarker_scores(data=data, cancer_type=cancer_type, use_case="diagnostic", limit=top_n)
    genes = ranked["gene_symbol"].tolist()
    pairs = []

    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            a = ranked.iloc[i]
            b = ranked.iloc[j]
            joint_specificity = min(float(a["tumor_specificity"]), float(b["tumor_specificity"]))
            pair_risk = float(a["off_target_risk"]) * float(b["off_target_risk"]) 
            pair_score = (0.5 * joint_specificity) - (0.3 * pair_risk) + (0.2 * max(float(a["ev_association"]), float(b["ev_association"])))
            pairs.append({
                "marker_a": a["gene_symbol"],
                "marker_b": b["gene_symbol"],
                "joint_tumor_specificity": joint_specificity,
                "joint_off_target_risk": pair_risk,
                "pair_score": pair_score,
            })

    return {
        "cancer_type": cancer_type,
        "top_combinations": sorted(pairs, key=lambda x: x["pair_score"], reverse=True)[:top_n],
    }


@app.get("/")
def root() -> dict:
    return {
        "message": "Cancer Biomarker Platform API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
