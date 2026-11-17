# 04_null_model_validation.py

from importlib.machinery import SourceFileLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

poc = SourceFileLoader("poc", "02_directed_tda_poc.py").load_module()

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def row_permute_offdiag(W_mat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = W_mat.shape[0]
    W_perm = W_mat.copy()
    for i in range(n):
        off_idx = [j for j in range(n) if j != i]
        vals = W_mat[i, off_idx].copy()
        rng.shuffle(vals)
        W_perm[i, off_idx] = vals
    return W_perm

def directed_degree_preserving_rewire(A: np.ndarray, rng: np.random.Generator, n_swaps: int = 150) -> np.ndarray:
    M = A.copy()
    np.fill_diagonal(M, 0)
    edges = list(zip(*np.where(M > 0)))
    m = len(edges)
    if m < 2:
        return M
    edge_set = set(edges)
    for _ in range(n_swaps):
        idx1, idx2 = rng.integers(0, m, size=2)
        if idx1 == idx2:
            continue
        u, v = edges[idx1]
        x, y = edges[idx2]
        if u == y or x == v or u == x or v == y:
            continue
        if (u, y) in edge_set or (x, v) in edge_set:
            continue
        edge_set.remove((u, v))
        edge_set.remove((x, y))
        edge_set.add((u, y))
        edge_set.add((x, v))
        edges[idx1] = (u, y)
        edges[idx2] = (x, v)

    M_new = np.zeros_like(M)
    for u, v in edges:
        M_new[u, v] = 1
    return M_new

def evaluate_matrix(W_mat: np.ndarray, thresholds: np.ndarray) -> dict:
    n = W_mat.shape[0]
    glmy_reg_sum = 0
    dfl_b1_sum = 0
    undir_b1_sum = 0
    sink_source_asym_20 = 0
    sink_source_asym_26 = 0

    for th in thresholds:
        A = (W_mat >= th).astype(int)
        uc = poc.build_undirected_clique_complex(A)
        uc_b = poc.compute_simplicial_betti(n, uc)
        dfl = poc.build_directed_flag_complex(A)
        dfl_b = poc.compute_directed_flag_betti(n, dfl)
        glmy_reg = poc.compute_glmy_path_homology(A, regular=True)

        undir_b1_sum += uc_b[1]
        dfl_b1_sum += dfl_b[1]
        glmy_reg_sum += glmy_reg["beta1"]

        if np.isclose(th, 0.20) or np.isclose(th, 0.26):
            sink = poc.build_dowker_complex(A, mode="sink")
            source = poc.build_dowker_complex(A, mode="source")
            asym = len(sink[2]) - len(source[2])
            if np.isclose(th, 0.20):
                sink_source_asym_20 = asym
            if np.isclose(th, 0.26):
                sink_source_asym_26 = asym

    return {
        "glmy_reg_b1_area": glmy_reg_sum,
        "dfl_b1_area": dfl_b1_sum,
        "undir_b1_area": undir_b1_sum,
        "sink_source_asym_20": sink_source_asym_20,
        "sink_source_asym_26": sink_source_asym_26,
    }

def main():
    W = pd.read_csv("output/tna_weights.csv", index_col=0)
    W_mat = W.values
    n = W_mat.shape[0]
    rng = np.random.default_rng(42)

    thresholds = np.round(np.arange(0.14, 0.301, 0.02), 2)
    obs = evaluate_matrix(W_mat, thresholds)

    B = 20000
    print(f"Running Null Model 1 (Row-wise entry permutation, B={B})...")
    df_null1 = pd.DataFrame([evaluate_matrix(row_permute_offdiag(W_mat, rng), thresholds) for _ in range(B)])

    print(f"Running Null Model 2 (Degree-preserving rewiring at th=0.26, B={B})...")
    A26 = (W_mat >= 0.26).astype(int)
    obs_glmy_reg26 = poc.compute_glmy_path_homology(A26, regular=True)["beta1"]

    null2_records = []
    for _ in range(B):
        A_rw = directed_degree_preserving_rewire(A26, rng, n_swaps=150)
        uc_b1 = poc.compute_simplicial_betti(n, poc.build_undirected_clique_complex(A_rw))[1]
        dfl_b1 = poc.compute_directed_flag_betti(n, poc.build_directed_flag_complex(A_rw))[1]
        reg_b1 = poc.compute_glmy_path_homology(A_rw, regular=True)["beta1"]
        null2_records.append({
            "glmy_reg_b1_26": reg_b1,
            "blind_spot_dfl_26": 1 if (uc_b1 == 0 and dfl_b1 >= 1) else 0,
        })
    df_null2 = pd.DataFrame(null2_records)

    def emp_p_upper(arr, val):
        return (1.0 + np.sum(arr >= val)) / (len(arr) + 1.0)

    def emp_p_lower(arr, val):
        return (1.0 + np.sum(arr <= val)) / (len(arr) + 1.0)

    summary_rows = [
        {"null_model": "Row-wise Entry Perm.", "metric": "Sink-Source asym (th=0.20)",
         "observed": obs["sink_source_asym_20"], "null_mean": df_null1["sink_source_asym_20"].mean(),
         "null_sd": df_null1["sink_source_asym_20"].std(), "p_value": emp_p_upper(df_null1["sink_source_asym_20"].values, obs["sink_source_asym_20"]), "tail": "upper"},
        {"null_model": "Row-wise Entry Perm.", "metric": "Sink-Source asym (th=0.26)",
         "observed": obs["sink_source_asym_26"], "null_mean": df_null1["sink_source_asym_26"].mean(),
         "null_sd": df_null1["sink_source_asym_26"].std(), "p_value": emp_p_upper(df_null1["sink_source_asym_26"].values, obs["sink_source_asym_26"]), "tail": "upper"},
        {"null_model": "Row-wise Entry Perm.", "metric": "GLMY beta1_path sum [0.14-0.30]",
         "observed": obs["glmy_reg_b1_area"], "null_mean": df_null1["glmy_reg_b1_area"].mean(),
         "null_sd": df_null1["glmy_reg_b1_area"].std(), "p_value": emp_p_lower(df_null1["glmy_reg_b1_area"].values, obs["glmy_reg_b1_area"]), "tail": "lower"},
        {"null_model": "Row-wise Entry Perm.", "metric": "dFl beta1 sum [0.14-0.30]",
         "observed": obs["dfl_b1_area"], "null_mean": df_null1["dfl_b1_area"].mean(),
         "null_sd": df_null1["dfl_b1_area"].std(), "p_value": emp_p_lower(df_null1["dfl_b1_area"].values, obs["dfl_b1_area"]), "tail": "lower"},
        {"null_model": "Degree-Preserving (th=0.26)", "metric": "GLMY beta1_path (th=0.26)",
         "observed": obs_glmy_reg26, "null_mean": df_null2["glmy_reg_b1_26"].mean(),
         "null_sd": df_null2["glmy_reg_b1_26"].std(), "p_value": emp_p_lower(df_null2["glmy_reg_b1_26"].values, obs_glmy_reg26), "tail": "lower"},
        {"null_model": "Degree-Preserving (th=0.26)", "metric": "Symmetrization blind spot (uc=0, dFl>=1)",
         "observed": 1, "null_mean": df_null2["blind_spot_dfl_26"].mean(),
         "null_sd": df_null2["blind_spot_dfl_26"].std(), "p_value": emp_p_upper(df_null2["blind_spot_dfl_26"].values, 1), "tail": "upper"},
    ]
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv("output/null_model_summary.csv", index=False)

    print("\n=== Null Model Validation Summary ===")
    print(df_sum.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    ax = axes[0]
    bins_asym = np.arange(-0.5, max(df_null1["sink_source_asym_26"].max(), obs["sink_source_asym_26"]) + 1.5, 1.0)
    counts0, _, _ = ax.hist(df_null1["sink_source_asym_26"], bins=bins_asym, color="#94A3B8", edgecolor="white", alpha=0.85,
                            label="Null 1 (Row-wise entry perm.)")
    ax.axvline(obs["sink_source_asym_26"], color="#D55E00", ls="--", lw=2.2,
               label=f"Observed = {obs['sink_source_asym_26']} ($p = {emp_p_upper(df_null1['sink_source_asym_26'].values, obs['sink_source_asym_26']):.3f}$)")
    ax.set_xlim(-1.5, 43.0)
    ax.set_ylim(0, max(counts0) * 1.28)
    ax.set_title("(a) Sink–Source Asymmetry Null Test ($\\tau=0.26$)")
    ax.set_xlabel("Sink 2-Simplices $-$ Source 2-Simplices")
    ax.set_ylabel("Frequency (20,000 permutations)")
    ax.legend(frameon=False, loc="upper right", fontsize=8.2)

    ax = axes[1]
    bins_b1 = np.arange(-0.5, max(df_null1["glmy_reg_b1_area"].max(), obs["glmy_reg_b1_area"]) + 1.5, 1.0)
    counts1, _, _ = ax.hist(df_null1["glmy_reg_b1_area"], bins=bins_b1, color="#94A3B8", edgecolor="white", alpha=0.85,
                            label="Null 1 (Row-wise entry perm.)")
    ax.axvline(obs["glmy_reg_b1_area"], color="#0072B2", ls="--", lw=2.2,
               label=f"Observed = {obs['glmy_reg_b1_area']} ($p_{{\\mathrm{{lower}}}} = {emp_p_lower(df_null1['glmy_reg_b1_area'].values, obs['glmy_reg_b1_area']):.3f}$)")
    ax.set_ylim(0, max(counts1) * 1.28)
    ax.set_title("(b) GLMY $\\beta_1^{\\mathrm{path}}$ Sum ($\\tau\\in[0.14, 0.30]$)")
    ax.set_xlabel("Sum of $\\beta_1^{\\mathrm{path}}$ across 9 thresholds")
    ax.set_ylabel("Frequency (20,000 permutations)")
    ax.legend(frameon=False, loc="upper right", fontsize=8.2)

    plt.tight_layout()
    fig.savefig("output/fig3_null_model_test.png", bbox_inches="tight")
    fig.savefig("output/fig3_null_model_test.pdf", bbox_inches="tight")
    plt.close(fig)
    print("\nSaved: output/null_model_summary.csv, output/fig3_null_model_test.png & .pdf")

if __name__ == "__main__":
    main()
