# Cancer Biomarker Platform

AI-powered biomedical evidence and decision support for translational oncology.

## What this project is

This platform turns heterogeneous biomedical data into explainable, prioritized biomarker recommendations. Instead of building a generic chatbot, the product is a decision engine that answers clinically useful questions such as:

- Which biomarkers best distinguish responders from non-responders?
- Which targets are tumor-specific and less likely to be off-target?
- Which biomarker combinations look promising for a diagnostic assay?
- What evidence is strongest, and what caveats remain?

## Why it matters

The project is designed to sit at the intersection of oncology data, ML, and clinical decision support. The same architecture can support multiple verticals:

- CRISPR / functional genomics
- single-cell analytics
- diagnostics and biomarker panels
- drug response interpretation
- liquid biopsy signatures
- proteomic evidence ranking
- clinical phenotype integration

## Core platform idea

Question → Evidence → Analysis → ML ranking → GenAI synthesis → Decision

Example:

"Find the strongest candidate biomarkers for early detection of Disease X."

The system can:
- query relevant datasets in a data lake,
- standardize and harmonize multiple input modalities,
- run statistical and ML analyses,
- retrieve supporting evidence from literature and internal data,
- rank candidates by confidence and risk tradeoffs,
- summarize with GenAI and provide caveats,
- recommend the next experiment.

## Architecture

```text
Biomedical data sources
  ├─ genomics
  ├─ transcriptomics
  ├─ single-cell
  ├─ proteomics
  ├─ clinical / diagnostics
  └─ drug-response
            ↓
       AWS data lake
            ↓
      Evidence layer
      ├─ statistical analysis
      ├─ ML ranking
      └─ knowledge retrieval
            ↓
      Biomarker engine
      ├─ ranking
      ├─ evidence synthesis
      └─ confidence scoring
            ↓
      GenAI decision layer
      ├─ explainable recommendation
      ├─ caveats
      └─ next-test suggestions
            ↓
      Scientist / leadership UI
```

AWS is the infrastructure. The product is the decision engine.

## MVP in this repo

This repository contains a working MVP scaffold that demonstrates:
- a biomedical biomarker scoring model,
- a sample dataset for cancer biomarker candidates,
- a FastAPI backend for candidate ranking,
- a small demo notebook for local exploration.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open:
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/api/v1/health
- http://127.0.0.1:8000/api/v1/candidates?cancer_type=PDAC&limit=5

## Exact validation commands

```bash
# 1) install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2) run the API locally
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# 3) health check
curl http://127.0.0.1:8000/api/v1/health

# 4) candidate query
curl "http://127.0.0.1:8000/api/v1/candidates?cancer_type=PDAC&limit=5"

# 5) optional list of cancer types
curl http://127.0.0.1:8000/api/v1/cancer-types
```

Expected health response:

```json
{
  "status": "healthy",
  "service": "cancer-biomarker-platform",
  "version": "0.1.0"
}
```

## Demo notebook

A small notebook is included in the `notebooks/` folder. It loads the sample biomarker dataset, ranks candidates, and shows the top biomarkers by cancer type.

## Project roadmap

1. swap the synthetic dataset for curated public evidence tables,
2. add retrieval for literature and external biomedical sources,
3. connect the backend to a decision-support UI,
4. add ML explainability and confidence calibration,
5. deploy on AWS with S3 + orchestration + GenAI layers.

## Repository intent

This project is positioned as a portfolio-grade oncology biomarker intelligence platform focused on evidence-backed clinical and scientific decision support.
