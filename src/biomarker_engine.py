from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "sample_biomarkers.csv"


def load_candidates(file_path: str | Path = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["tumor_normal_ratio"] = (
        df["tumor_tpm"] / (df["normal_tpm"] + 1e-6)
    ).replace([np.inf, -np.inf], np.nan)
    df["tumor_normal_ratio"] = df["tumor_normal_ratio"].fillna(0.0)
    return df


def score_candidate(row: pd.Series) -> float:
    specificity = float(row["specificity_score"])
    off_target = 1.0 - float(row["off_target_risk"])
    accessibility = 1.0 if bool(row["surface_accessible"]) else 0.5
    response_signal = float(row["response_signal"])
    evidence_strength = float(row["evidence_strength"])

    score = (
        0.35 * specificity
        + 0.20 * off_target
        + 0.20 * accessibility
        + 0.15 * response_signal
        + 0.10 * evidence_strength
    )
    return float(np.clip(score, 0.0, 1.0))


def rank_candidates(cancer_type: str | None = None, limit: int = 10) -> pd.DataFrame:
    df = load_candidates()
    if cancer_type:
        df = df[df["cancer_type"].str.upper() == cancer_type.upper()]

    if df.empty:
        return pd.DataFrame(columns=[
            "gene_symbol",
            "cancer_type",
            "tumor_tpm",
            "normal_tpm",
            "surface_accessible",
            "specificity_score",
            "off_target_risk",
            "response_signal",
            "evidence_strength",
            "tumor_normal_ratio",
            "priority_score",
            "rank",
        ])

    df = df.copy()
    df["priority_score"] = df.apply(score_candidate, axis=1)
    df = df.sort_values(["priority_score", "tumor_normal_ratio"], ascending=[False, False]).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df.head(limit)


def list_cancer_types() -> list[str]:
    df = load_candidates()
    return sorted(df["cancer_type"].unique().tolist())


if __name__ == "__main__":
    for cancer in ["PDAC", "BRCA"]:
        print(f"\nTop biomarker candidates for {cancer}:\n")
        ranked = rank_candidates(cancer, limit=5)
        print(ranked[["rank", "gene_symbol", "priority_score", "tumor_normal_ratio", "surface_accessible"]].to_string(index=False))
