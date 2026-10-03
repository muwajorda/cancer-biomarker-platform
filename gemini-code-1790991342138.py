#!/usr/bin/env python3
"""
Mercy Bio Analytics - Biomarker Prioritization & Multi-Omics Data Engine
=======================================================================
A robust framework for integrating public cancer resources (TCGA, GTEx,
CPTAC, DepMap, Human Protein Atlas, Single-Cell Atlases), processing internal cell
line/clinical assay profiles, and executing single- and dual-marker prioritization.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CancerDataPlatform")


# ==============================================================================
# 1. CONFIGURATION & SCORING WEIGHTS
# ==============================================================================

@dataclass
class PrioritizationWeights:
    """Weighted contribution of each evidence layer in the final biomarker score."""

    tumor_specificity: float = 0.35
    off_target_penalty: float = 0.25
    surface_accessibility: float = 0.15
    vesicle_association: float = 0.15
    reagent_availability: float = 0.10

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative; got {value!r}")

        total = (
            self.tumor_specificity
            + self.off_target_penalty
            + self.surface_accessibility
            + self.vesicle_association
            + self.reagent_availability
        )
        if not np.isclose(total, 1.0):
            logger.warning(
                "Weight total is %.3f instead of 1.0. Results are still valid but may be intentionally tuned.",
                total,
            )


# ==============================================================================
# 2. INTERNAL DATA QC & HIT CALLING
# ==============================================================================

class InternalDataProcessor:
    """Handles plate-level diagnostics, normalization, batch effect removal, and hit calling."""

    @staticmethod
    def normalize_plate_data(df: pd.DataFrame, signal_col: str, plate_col: str) -> pd.DataFrame:
        """Apply median normalization per plate to account for inter-plate variability."""
        if signal_col not in df.columns:
            raise ValueError(f"Signal column '{signal_col}' not found in dataframe.")
        if plate_col not in df.columns:
            raise ValueError(f"Plate column '{plate_col}' not found in dataframe.")

        df = df.copy()
        plate_medians = df.groupby(plate_col)[signal_col].transform("median")
        df["normalized_signal"] = df[signal_col] / (plate_medians + 1e-8)
        df["log2_normalized_signal"] = np.log2(df["normalized_signal"] + 1e-4)
        logger.info("Executed median plate-level normalization across %d plates.", df[plate_col].nunique())
        return df

    @staticmethod
    def compute_zscores(df: pd.DataFrame, signal_col: str, group_col: Optional[str] = None) -> pd.DataFrame:
        """Calculate robust Z-scores (median and MAD) for hit calling."""
        if signal_col not in df.columns:
            raise ValueError(f"Signal column '{signal_col}' not found in dataframe.")

        df = df.copy()
        if group_col:
            if group_col not in df.columns:
                raise ValueError(f"Grouping column '{group_col}' not found in dataframe.")
            median = df.groupby(group_col)[signal_col].transform("median")
            mad = df.groupby(group_col)[signal_col].transform(
                lambda x: np.median(np.abs(x - np.median(x)))
            )
        else:
            median = df[signal_col].median()
            mad = np.median(np.abs(df[signal_col] - median))

        df["robust_zscore"] = (df[signal_col] - median) / (1.4826 * mad + 1e-8)
        return df

    @staticmethod
    def call_hits(df: pd.DataFrame, z_threshold: float = 2.5, fold_change_threshold: float = 1.5) -> pd.DataFrame:
        """Identify significant hits based on Z-score and fold-change thresholds."""
        required_cols = {"robust_zscore", "normalized_signal"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for hit calling: {sorted(missing)}")

        df = df.copy()
        df["is_hit"] = (df["robust_zscore"] >= z_threshold) & (df["normalized_signal"] >= fold_change_threshold)
        logger.info("Hit calling complete: Identified %d hits out of %d samples.", int(df["is_hit"].sum()), len(df))
        return df


# ==============================================================================
# 3. PUBLIC DATA INTEGRATOR & EVIDENCE LAYERS
# ==============================================================================

class PublicDataIntegrator:
    """Generate or simulate public evidence layers for cancer biomarker prioritization."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.evidence_db = pd.DataFrame()

    def load_mock_evidence_layers(self, gene_list: Iterable[str]) -> pd.DataFrame:
        """Generate a structured evidence matrix for target candidates."""
        genes = [str(gene).strip() for gene in gene_list if str(gene).strip()]
        if not genes:
            raise ValueError("gene_list must contain at least one gene symbol.")

        np.random.seed(self.seed)
        surface_locations = ["Plasma membrane", "Extracellular", "Cytoplasm", "Nucleus", "Mitochondria"]

        rows = []
        for gene in genes:
            tcga_mean = np.random.uniform(2.0, 10.0)
            gtex_mean = np.random.uniform(0.1, 8.0)
            gtex_max_essential = np.random.uniform(0.1, 9.0)
            cptac_protein_corr = np.random.uniform(-0.2, 0.9)
            hpa_location = np.random.choice(surface_locations, p=[0.3, 0.2, 0.2, 0.2, 0.1])
            ev_association_score = np.random.uniform(0.0, 1.0)
            reagent_score = np.random.choice([0.0, 0.5, 1.0], p=[0.2, 0.3, 0.5])
            sc_tumor_fraction = np.random.uniform(0.05, 0.95)

            rows.append(
                {
                    "gene_symbol": gene,
                    "tcga_exp_log2": tcga_mean,
                    "gtex_exp_log2": gtex_mean,
                    "gtex_max_essential_organ": gtex_max_essential,
                    "cptac_protein_rna_corr": cptac_protein_corr,
                    "hpa_subcellular_loc": hpa_location,
                    "ev_association_score": ev_association_score,
                    "reagent_availability": reagent_score,
                    "sc_tumor_pos_fraction": sc_tumor_fraction,
                }
            )

        self.evidence_db = pd.DataFrame(rows)
        logger.info("Loaded evidence layers for %d genes.", len(genes))
        return self.evidence_db


# ==============================================================================
# 4. BIOMARKER PRIORITIZATION FRAMEWORK (SINGLE & PAIRWISE)
# ==============================================================================

class BiomarkerPrioritizer:
    """Prioritize biomarker candidates based on multi-dimensional evidence."""

    def __init__(self, weights: Optional[PrioritizationWeights] = None):
        self.weights = weights or PrioritizationWeights()

    @staticmethod
    def _normalize_to_unit(values: pd.Series) -> pd.Series:
        min_v = values.min()
        max_v = values.max()
        span = max_v - min_v
        if np.isclose(span, 0.0):
            return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
        return (values - min_v) / span

    def calculate_single_target_scores(self, evidence_df: pd.DataFrame) -> pd.DataFrame:
        """Score and rank individual target candidates."""
        if evidence_df.empty:
            raise ValueError("Evidence dataframe is empty.")

        df = evidence_df.copy()

        # 1. Tumor specificity score (TCGA vs GTEx delta)
        df["delta_tcga_gtex"] = df["tcga_exp_log2"] - df["gtex_exp_log2"]
        df["score_tumor_specificity"] = np.clip(df["delta_tcga_gtex"] / 5.0, 0, 1)

        # 2. Off-target risk score (higher = worse)
        df["score_off_target_penalty"] = np.clip(df["gtex_max_essential_organ"] / 8.0, 0, 1)

        # 3. Surface accessibility score
        df["score_surface_accessibility"] = df["hpa_subcellular_loc"].apply(
            lambda loc: 1.0 if loc in ["Plasma membrane", "Extracellular"] else 0.2
        )

        # 4. Vesicle association score
        df["score_vesicle"] = df["ev_association_score"]

        # 5. Reagent availability score
        df["score_reagent"] = df["reagent_availability"]

        df["composite_score"] = (
            self.weights.tumor_specificity * df["score_tumor_specificity"]
            - self.weights.off_target_penalty * df["score_off_target_penalty"]
            + self.weights.surface_accessibility * df["score_surface_accessibility"]
            + self.weights.vesicle_association * df["score_vesicle"]
            + self.weights.reagent_availability * df["score_reagent"]
        )

        df["prioritization_score"] = 100.0 * self._normalize_to_unit(df["composite_score"])
        return df.sort_values(by="prioritization_score", ascending=False).reset_index(drop=True)

    def prioritize_pairs(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Evaluate combinatorial pairs for logic-gated assay designs."""
        if df.empty:
            raise ValueError("Input dataframe is empty.")

        top_candidates = df.head(max(2, top_n))["gene_symbol"].tolist()
        pairs = []

        for i in range(len(top_candidates)):
            for j in range(i + 1, len(top_candidates)):
                g1, g2 = top_candidates[i], top_candidates[j]
                row1 = df[df["gene_symbol"] == g1].iloc[0]
                row2 = df[df["gene_symbol"] == g2].iloc[0]

                joint_tumor_coverage = min(row1["sc_tumor_pos_fraction"], row2["sc_tumor_pos_fraction"])
                joint_off_target_risk = row1["score_off_target_penalty"] * row2["score_off_target_penalty"]
                pair_surface_score = (row1["score_surface_accessibility"] + row2["score_surface_accessibility"]) / 2.0
                pair_ev_score = (row1["score_vesicle"] + row2["score_vesicle"]) / 2.0

                pair_composite = (
                    0.4 * joint_tumor_coverage
                    - 0.3 * joint_off_target_risk
                    + 0.15 * pair_surface_score
                    + 0.15 * pair_ev_score
                )

                pairs.append(
                    {
                        "target_A": g1,
                        "target_B": g2,
                        "joint_tumor_coverage": joint_tumor_coverage,
                        "joint_off_target_risk": joint_off_target_risk,
                        "pair_prioritization_score": pair_composite,
                    }
                )

        if not pairs:
            return pd.DataFrame(
                columns=[
                    "target_A",
                    "target_B",
                    "joint_tumor_coverage",
                    "joint_off_target_risk",
                    "pair_prioritization_score",
                ]
            )

        pair_df = pd.DataFrame(pairs)
        return pair_df.sort_values(by="pair_prioritization_score", ascending=False).reset_index(drop=True)


# ==============================================================================
# 5. DATA WAREHOUSE & EXPORT MANAGEMENT
# ==============================================================================

class DataWarehouseManager:
    """Manage storage and export of queryable evidence layers and candidate rankings."""

    def __init__(self, output_dir: str = "./warehouse_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write_dataframe(self, df: pd.DataFrame, name: str) -> Path:
        target = self.output_dir / name
        try:
            df.to_parquet(target, index=False)
            logger.info("Saved Parquet export: %s", target)
            return target
        except ImportError:
            csv_path = target.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            logger.warning("pyarrow not installed; wrote CSV fallback to %s", csv_path)
            return csv_path

    def export_results(self, single_df: pd.DataFrame, pair_df: pd.DataFrame) -> dict:
        """Save analysis results to Parquet/CSV and JSON files for downstream dashboards."""
        single_path = self._write_dataframe(single_df, "single_target_prioritization.parquet")
        pair_path = self._write_dataframe(pair_df, "pairwise_target_prioritization.parquet")
        summary_path = self.output_dir / "prioritization_summary.json"

        summary = {
            "total_candidates_evaluated": len(single_df),
            "top_5_single_targets": single_df.head(5)["gene_symbol"].tolist(),
            "top_3_target_pairs": pair_df.head(3)[["target_A", "target_B"]].to_dict(orient="records"),
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        logger.info("Successfully exported evidence layers and prioritization reports to %s", self.output_dir)
        return {
            "single_path": str(single_path),
            "pair_path": str(pair_path),
            "summary_path": str(summary_path),
        }


# ==============================================================================
# 6. MAIN EXECUTION PIPELINE
# ==============================================================================

def build_candidate_genes() -> List[str]:
    return [
        "EGFR", "ERBB2", "MUC1", "EPCAM", "CEACAM5", "FOLR1", "VTCN1",
        "MSLN", "TROP2", "MET", "CD24", "CD44", "PROM1", "AXL", "GPC3"
    ]


def run_pipeline(candidate_genes: Optional[List[str]] = None, output_dir: str = "./warehouse_outputs") -> pd.DataFrame:
    candidate_genes = candidate_genes or build_candidate_genes()

    integrator = PublicDataIntegrator(seed=42)
    evidence_df = integrator.load_mock_evidence_layers(candidate_genes)

    prioritizer = BiomarkerPrioritizer()
    single_ranked_df = prioritizer.calculate_single_target_scores(evidence_df)
    pair_ranked_df = prioritizer.prioritize_pairs(single_ranked_df, top_n=10)

    mock_assay_data = pd.DataFrame(
        {
            "sample_id": [f"S_{i}" for i in range(100)],
            "plate_id": [f"Plate_{i // 25 + 1}" for i in range(100)],
            "raw_signal": np.random.normal(loc=1000, scale=150, size=100)
            + np.random.choice([0, 800], p=[0.9, 0.1], size=100),
        }
    )

    proc_df = InternalDataProcessor.normalize_plate_data(mock_assay_data, "raw_signal", "plate_id")
    proc_df = InternalDataProcessor.compute_zscores(proc_df, "normalized_signal", "plate_id")
    _ = InternalDataProcessor.call_hits(proc_df)

    warehouse = DataWarehouseManager(output_dir=output_dir)
    warehouse.export_results(single_ranked_df, pair_ranked_df)

    return single_ranked_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prioritize biomarker candidates using public evidence layers.")
    parser.add_argument("--genes", type=str, default=None, help="Comma-separated gene list to evaluate.")
    parser.add_argument("--output-dir", type=str, default="./warehouse_outputs", help="Directory for exports.")
    args = parser.parse_args()

    print("=" * 80)
    print(" Mercy Bio - Biomarker Prioritization & Data Platform Engine")
    print("=" * 80)

    candidates = None
    if args.genes:
        candidates = [gene.strip() for gene in args.genes.split(",") if gene.strip()]

    single_ranked_df = run_pipeline(candidate_genes=candidates, output_dir=args.output_dir)

    print("\n--- TOP 5 SINGLE BIOMARKER CANDIDATES ---")
    print(
        single_ranked_df[
            ["gene_symbol", "prioritization_score", "tcga_exp_log2", "gtex_max_essential_organ", "hpa_subcellular_loc"]
        ]
        .head(5)
        .to_string(index=False)
    )

    pair_ranked_df = BiomarkerPrioritizer().prioritize_pairs(single_ranked_df, top_n=10)
    print("\n--- TOP 5 DUAL-MARKER COMBINATIONS (LOGIC GATES) ---")
    print(
        pair_ranked_df[
            ["target_A", "target_B", "joint_tumor_coverage", "joint_off_target_risk", "pair_prioritization_score"]
        ]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
