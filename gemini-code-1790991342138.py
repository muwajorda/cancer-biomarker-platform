#!/usr/bin/env python3
"""
Mercy Bio Analytics - Biomarker Prioritization & Multi-Omics Data Engine
=======================================================================
A comprehensive framework for integrating public cancer resources (TCGA, GTEx,
CPTAC, DepMap, Human Protein Atlas, Single-Cell Atlases), processing internal cell
line/clinical assay profiles, and executing single- and dual-marker prioritization.
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CancerDataPlatform")


# ==============================================================================
# 1. CONFIGURATION & SCORING WEIGHTS
# ==============================================================================

@dataclass
class PrioritizationWeights:
    tumor_specificity: float = 0.35
    off_target_penalty: float = 0.25
    surface_accessibility: float = 0.15
    vesicle_association: float = 0.15
    reagent_availability: float = 0.10


# ==============================================================================
# 2. INTERNAL DATA QC & HIT CALLING
# ==============================================================================

class InternalDataProcessor:
    """Handles plate-level diagnostics, normalization, batch effect removal, and hit calling."""

    @staticmethod
    def normalize_plate_data(df: pd.DataFrame, signal_col: str, plate_col: str) -> pd.DataFrame:
        """Applies median normalization per plate to account for inter-plate variability."""
        df = df.copy()
        plate_medians = df.groupby(plate_col)[signal_col].transform('median')
        df['normalized_signal'] = df[signal_col] / (plate_medians + 1e-8)
        df['log2_normalized_signal'] = np.log2(df['normalized_signal'] + 1e-4)
        logger.info("Executed median plate-level normalization across %d plates.", df[plate_col].nunique())
        return df

    @staticmethod
    def compute_zscores(df: pd.DataFrame, signal_col: str, group_col: Optional[str] = None) -> pd.DataFrame:
        """Calculates robust Z-scores (using median and MAD) for hit calling."""
        df = df.copy()
        if group_col:
            median = df.groupby(group_col)[signal_col].transform('median')
            mad = df.groupby(group_col)[signal_col].transform(lambda x: np.median(np.abs(x - np.median(x))))
        else:
            median = df[signal_col].median()
            mad = np.median(np.abs(df[signal_col] - median))

        df['robust_zscore'] = (df[signal_col] - median) / (1.4826 * mad + 1e-8)
        return df

    @staticmethod
    def call_hits(df: pd.DataFrame, z_threshold: float = 2.5, fold_change_threshold: float = 1.5) -> pd.DataFrame:
        """Identifies significant hits based on Z-score and fold-change thresholds."""
        df['is_hit'] = (df['robust_zscore'] >= z_threshold) & (df['normalized_signal'] >= fold_change_threshold)
        logger.info("Hit calling complete: Identified %d hits out of %d samples.", df['is_hit'].sum(), len(df))
        return df


# ==============================================================================
# 3. PUBLIC DATA INTEGRATOR & EVIDENCE LAYERS
# ==============================================================================

class PublicDataIntegrator:
    """Simulates/Integrates public evidence layers (TCGA, GTEx, CPTAC, HPA, DepMap, Single-Cell)."""

    def __init__(self):
        self.evidence_db = pd.DataFrame()

    def load_mock_evidence_layers(self, gene_list: List[str]) -> pd.DataFrame:
        """Generates structured evidence layer matrix for target candidates."""
        np.random.seed(42)
        
        data = []
        surface_locations = ["Plasma membrane", "Extracellular", "Cytoplasm", "Nucleus", "Mitochondria"]
        
        for gene in gene_list:
            tcga_mean = np.random.uniform(2.0, 10.0)
            gtex_mean = np.random.uniform(0.1, 8.0)
            gtex_max_essential = np.random.uniform(0.1, 9.0)  # High in heart/brain/liver
            cptac_protein_corr = np.random.uniform(-0.2, 0.9)
            hpa_location = np.random.choice(surface_locations, p=[0.3, 0.2, 0.2, 0.2, 0.1])
            ev_association_score = np.random.uniform(0.0, 1.0) # Extracellular vesicle enrichment
            reagent_score = np.random.choice([0.0, 0.5, 1.0], p=[0.2, 0.3, 0.5]) # Antibody/binder availability
            sc_tumor_fraction = np.random.uniform(0.05, 0.95) # Fraction of tumor cells expressing gene
            
            data.append({
                "gene_symbol": gene,
                "tcga_exp_log2": tcga_mean,
                "gtex_exp_log2": gtex_mean,
                "gtex_max_essential_organ": gtex_max_essential,
                "cptac_protein_rna_corr": cptac_protein_corr,
                "hpa_subcellular_loc": hpa_location,
                "ev_association_score": ev_association_score,
                "reagent_availability": reagent_score,
                "sc_tumor_pos_fraction": sc_tumor_fraction
            })

        self.evidence_db = pd.DataFrame(data)
        logger.info("Loaded evidence layers for %d genes.", len(gene_list))
        return self.evidence_db


# ==============================================================================
# 4. BIOMARKER PRIORITIZATION FRAMEWORK (SINGLE & PAIRWISE)
# ==============================================================================

class BiomarkerPrioritizer:
    """Prioritizes biomarker candidates based on multi-dimensional evidence layers."""

    def __init__(self, weights: PrioritizationWeights = PrioritizationWeights()):
        self.weights = weights

    def calculate_single_target_scores(self, evidence_df: pd.DataFrame) -> pd.DataFrame:
        """Scores and ranks individual target candidates."""
        df = evidence_df.copy()

        # 1. Tumor Specificity Score (TCGA vs GTEx Delta)
        df['delta_tcga_gtex'] = df['tcga_exp_log2'] - df['gtex_exp_log2']
        df['score_tumor_specificity'] = np.clip(df['delta_tcga_gtex'] / 5.0, 0, 1)

        # 2. Off-Target Penalty (High expression in essential healthy organs)
        df['score_off_target_penalty'] = np.clip(df['gtex_max_essential_organ'] / 8.0, 0, 1)

        # 3. Surface Accessibility
        df['score_surface_accessibility'] = df['hpa_subcellular_loc'].apply(
            lambda loc: 1.0 if loc in ["Plasma membrane", "Extracellular"] else 0.2
        )

        # 4. Vesicle Association
        df['score_vesicle'] = df['ev_association_score']

        # 5. Reagent Availability
        df['score_reagent'] = df['reagent_availability']

        # Composite Score Calculation
        df['composite_score'] = (
            self.weights.tumor_specificity * df['score_tumor_specificity'] -
            self.weights.off_target_penalty * df['score_off_target_penalty'] +
            self.weights.surface_accessibility * df['score_surface_accessibility'] +
            self.weights.vesicle_association * df['score_vesicle'] +
            self.weights.reagent_availability * df['score_reagent']
        )

        # Normalize score to [0, 100]
        min_s, max_s = df['composite_score'].min(), df['composite_score'].max()
        df['prioritization_score'] = 100 * (df['composite_score'] - min_s) / (max_s - min_s + 1e-8)
        
        return df.sort_values(by="prioritization_score", ascending=False)

    def prioritize_pairs(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Evaluates biomarker pairs for co-expression / logic gate assay designs (Target A AND Target B).
        Reduces off-target toxicity while maintaining high tumor coverage.
        """
        top_candidates = df.head(top_n)['gene_symbol'].tolist()
        pairs = []

        for i in range(len(top_candidates)):
            for j in range(i + 1, len(top_candidates)):
                g1, g2 = top_candidates[i], top_candidates[j]
                row1 = df[df['gene_symbol'] == g1].iloc[0]
                row2 = df[df['gene_symbol'] == g2].iloc[0]

                # Co-expression logic score (AND gate): Tumor specificity improves, off-target risk drops
                joint_tumor_coverage = min(row1['sc_tumor_pos_fraction'], row2['sc_tumor_pos_fraction'])
                joint_off_target_risk = row1['score_off_target_penalty'] * row2['score_off_target_penalty']
                pair_surface_score = (row1['score_surface_accessibility'] + row2['score_surface_accessibility']) / 2.0
                pair_ev_score = (row1['score_vesicle'] + row2['score_vesicle']) / 2.0

                pair_composite = (
                    0.4 * joint_tumor_coverage -
                    0.3 * joint_off_target_risk +
                    0.15 * pair_surface_score +
                    0.15 * pair_ev_score
                )

                pairs.append({
                    "target_A": g1,
                    "target_B": g2,
                    "joint_tumor_coverage": joint_tumor_coverage,
                    "joint_off_target_risk": joint_off_target_risk,
                    "pair_prioritization_score": pair_composite
                })

        pair_df = pd.DataFrame(pairs)
        return pair_df.sort_values(by="pair_prioritization_score", ascending=False)


# ==============================================================================
# 5. DATA WAREHOUSE & EXPORT MANAGEMENT
# ==============================================================================

class DataWarehouseManager:
    """Manages storage and export of queryable evidence layers and candidate rankings."""

    def __init__(self, output_dir: str = "./warehouse_outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export_results(self, single_df: pd.DataFrame, pair_df: pd.DataFrame):
        """Saves analysis results to Parquet and JSON files for downstream dashboards."""
        single_path = os.path.join(self.output_dir, "single_target_prioritization.parquet")
        pair_path = os.path.join(self.output_dir, "pairwise_target_prioritization.parquet")
        summary_path = os.path.join(self.output_dir, "prioritization_summary.json")

        single_df.to_parquet(single_path, index=False)
        pair_df.to_parquet(pair_path, index=False)

        summary = {
            "total_candidates_evaluated": len(single_df),
            "top_5_single_targets": single_df.head(5)['gene_symbol'].tolist(),
            "top_3_target_pairs": pair_df.head(3)[['target_A', 'target_B']].to_dict(orient="records")
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        logger.info("Successfully exported evidence layers and prioritization reports to %s", self.output_dir)


# ==============================================================================
# 6. MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    print("=" * 80)
    print(" Mercy Bio - Biomarker Prioritization & Data Platform Engine")
    print("=" * 80)

    # 1. Target Candidate Pool
    candidate_genes = [
        "EGFR", "ERBB2", "MUC1", "EPCAM", "CEACAM5", "FOLR1", "VTCN1",
        "MSLN", "TROP2", "MET", "CD24", "CD44", "PROM1", "AXL", "GPC3"
    ]

    # 2. Public Data Integration Layer
    integrator = PublicDataIntegrator()
    evidence_df = integrator.load_mock_evidence_layers(candidate_genes)

    # 3. Target Prioritization
    prioritizer = BiomarkerPrioritizer()
    single_ranked_df = prioritizer.calculate_single_target_scores(evidence_df)
    pair_ranked_df = prioritizer.prioritize_pairs(single_ranked_df, top_n=10)

    # 4. Internal Assay Profiling & Hit Calling
    print("\n--- Processing Internal Screening Data ---")
    mock_assay_data = pd.DataFrame({
        "sample_id": [f"S_{i}" for i in range(100)],
        "plate_id": [f"Plate_{i//25 + 1}" for i in range(100)],
        "raw_signal": np.random.normal(loc=1000, scale=150, size=100) + np.random.choice([0, 800], p=[0.9, 0.1], size=100)
    })
    
    proc_df = InternalDataProcessor.normalize_plate_data(mock_assay_data, "raw_signal", "plate_id")
    proc_df = InternalDataProcessor.compute_zscores(proc_df, "normalized_signal", "plate_id")
    hits_df = InternalDataProcessor.call_hits(proc_df)

    # 5. Data Warehouse Export
    warehouse = DataWarehouseManager()
    warehouse.export_results(single_ranked_df, pair_ranked_df)

    # 6. Output Summaries
    print("\n--- TOP 5 SINGLE BIOMARKER CANDIDATES ---")
    print(single_ranked_df[['gene_symbol', 'prioritization_score', 'tcga_exp_log2', 'gtex_max_essential_organ', 'hpa_subcellular_loc']].head(5).to_string(index=False))

    print("\n--- TOP 5 DUAL-MARKER COMBINATIONS (LOGIC GATES) ---")
    print(pair_ranked_df[['target_A', 'target_B', 'joint_tumor_coverage', 'joint_off_target_risk', 'pair_prioritization_score']].head(5).to_string(index=False))

if __name__ == "__main__":
    main()