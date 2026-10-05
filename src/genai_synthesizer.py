"""Simple GenAI synthesizer using OpenAI (optional) with a fallback summarizer.

This module implements a small retrieval+prompt pipeline suitable for demos.
"""
import os
from typing import List

try:
    import openai
except Exception:
    openai = None


def _format_candidates_for_prompt(candidates: List[dict]) -> str:
    parts = []
    for c in candidates:
        parts.append(
            f"- {c.get('gene_symbol', c.get('gene', 'unknown'))} (score={c.get('priority_score', c.get('score', 'N/A')):.2f}): tumor_normal_ratio={c.get('tumor_normal_ratio', 'N/A')}, surface={c.get('surface_accessible', c.get('surface', False))}, evidence_strength={c.get('evidence_strength', 'N/A')}"
        )
    return "\n".join(parts)


def synthesize_decision(candidates: List[dict], question: str | None = None) -> dict:
    """Return a structured GenAI synthesis for a list of candidate dicts.

    If OPENAI_API_KEY is set and openai is available, call the chat completions API. Otherwise return a deterministic template.
    """
    prompt_intro = (
        "You are an expert scientific assistant. Given biomarker candidates and supporting evidence, produce a concise evidence-backed recommendation, list caveats, and suggest next experiments."
    )

    if question:
        prompt_intro += f" The user question: {question}"

    candidates_text = _format_candidates_for_prompt(candidates)

    system_prompt = prompt_intro + "\n\nCandidates:\n" + candidates_text

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and openai:
        openai.api_key = api_key
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Provide a structured JSON with fields: summary, top_candidate, caveats, recommended_experiments."},
                ],
                temperature=0.0,
                max_tokens=600,
            )
            text = resp.choices[0].message.content
            return {"model_response": text}
        except Exception as e:
            return {"error": "OpenAI call failed", "exception": str(e), "fallback": True}

    # Fallback deterministic synthesis
    if not candidates:
        return {
            "summary": "No candidates available.",
            "top_candidate": None,
            "caveats": [],
            "recommended_experiments": [],
        }

    top = sorted(candidates, key=lambda x: float(x.get("priority_score", x.get("score", 0))), reverse=True)[0]
    summary = f"Top candidate is {top.get('gene_symbol', top.get('gene'))} with priority {top.get('priority_score', top.get('score')):.2f}. It shows high tumor-normal ratio and supporting evidence."
    caveats = ["Limited external validation", "Protein-level evidence incomplete"]
    experiments = ["Prospective assay in early-stage cohort (n=50)", "Orthogonal protein validation (mass spec / ELISA)"]

    return {
        "summary": summary,
        "top_candidate": top,
        "caveats": caveats,
        "recommended_experiments": experiments,
    }
