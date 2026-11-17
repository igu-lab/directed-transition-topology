# Preserving Orientation and Transition Order in Transition Network Analysis

Replication code for:
**Preserving Orientation and Transition Order in Transition Network Analysis: A Directed Topological Proof of Concept**  
Koichi Yasutake (Hiroshima University, Japan) and Hitoshi Inoue (Nakamura Gakuen University, Japan).

## Files and Pipeline Overview

| Script | Language | Purpose & Outputs |
| --- | --- | --- |
| `01_tna_nestimate_baseline.R` | R | Estimates the TNA transition probability matrix $W$ on `group_regulation`, runs `tna::cliques()`, `Nestimate::betti_numbers()`, and `Nestimate::persistent_homology()` baselines, and exports `output/tna_weights.csv`, `output/tna_cliques_summary.csv`, `output/tna_boot_sig.csv`, and `output/nestimate_betti.csv`. |
| `02_directed_tda_poc.py` | Python | Reads `output/tna_weights.csv` and computes filtrations across $\tau \in [0.02, 0.30]$ (step `0.02`) for four directed constructions—Dowker sink, Dowker source, directed flag ($dFl$), and regular GLMY path homology—alongside the max-symmetrized clique baseline. Exports `output/poc_filtration_comparison.csv` and `output/poc_betti_curves.png`. |
| `03_extract_motifs_and_figures.py` | Python | Extracts simple directed cycles and degree profiles at $\tau = 0.26$ and generates **Figure 1** (`output/fig1_filtration_asymmetry.{png,pdf}`) and **Figure 2** (`output/fig2_mechanism_comparison.{png,pdf}`). |
| `04_null_model_validation.py` | Python | Runs row-wise entry permutation and degree-preserving directed rewiring ($B = 20{,}000$, NumPy `default_rng(42)`), reproducing **Table 1** (`output/null_model_summary.csv`) and **Figure 3** (`output/fig3_null_model_test.{png,pdf}`). |
| `05_bootstrap_stability.R` | R | Runs sequence-level bootstrap resampling ($B = 1{,}000$, `set.seed(42)`) over the 2,000 `group_regulation` sequences to reproduce the stability percentages reported in Section 3.1 (reciprocal loop, sink--source asymmetry, symmetrization blind spot at $\tau = 0.26$ and $\tau = 0.24$, unfilled 3-cycle at $\tau = 0.16$, and edge-level fluctuations). |

## Environment
- **R**: `tna` (v1.2.3), `Nestimate` (v1.0.12)
- **Python**: Python 3.10+, `numpy`, `pandas`, `matplotlib`

## Execution Order
Run from the repository root (`output/` will be created automatically):

```bash
Rscript 01_tna_nestimate_baseline.R
python3 02_directed_tda_poc.py
python3 03_extract_motifs_and_figures.py
python3 04_null_model_validation.py
Rscript 05_bootstrap_stability.R
```

## License
MIT License.
