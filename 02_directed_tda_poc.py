# 02_directed_tda_poc.py

from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Simplicial Homology (over R via SVD rank)
# ------------------------------------------------------------
def compute_simplicial_betti(n_nodes: int, simplices_by_dim: dict[int, list[tuple[int, ...]]]) -> dict[int, int]:
    s0 = [(i,) for i in range(n_nodes)]
    s1 = sorted(set(simplices_by_dim.get(1, [])))
    s2 = sorted(set(simplices_by_dim.get(2, [])))
    s3 = sorted(set(simplices_by_dim.get(3, [])))

    def boundary_matrix(high_s, low_s):
        if not high_s or not low_s:
            return np.zeros((len(low_s), len(high_s)), dtype=float)
        low_idx = {s: i for i, s in enumerate(low_s)}
        mat = np.zeros((len(low_s), len(high_s)), dtype=float)
        for j, sigma in enumerate(high_s):
            for k in range(len(sigma)):
                face = sigma[:k] + sigma[k + 1 :]
                if face in low_idx:
                    mat[low_idx[face], j] += (-1.0) ** k
        return mat

    def rank(mat):
        return int(np.linalg.matrix_rank(mat, tol=1e-9)) if mat.size else 0

    r1 = rank(boundary_matrix(s1, s0))
    r2 = rank(boundary_matrix(s2, s1))
    r3 = rank(boundary_matrix(s3, s2))
    return {
        0: len(s0) - r1,
        1: (len(s1) - r1) - r2,
        2: (len(s2) - r2) - r3,
    }

# ------------------------------------------------------------
# 2. Complex Builders from Directed Adjacency Matrix A (0/1)
# ------------------------------------------------------------
def build_undirected_clique_complex(A: np.ndarray) -> dict[int, list[tuple[int, ...]]]:
    n = A.shape[0]
    U = ((A + A.T) > 0).astype(int)
    np.fill_diagonal(U, 0)
    s1 = [(i, j) for i, j in combinations(range(n), 2) if U[i, j]]
    s2 = [c for c in combinations(range(n), 3) if U[c[0], c[1]] and U[c[0], c[2]] and U[c[1], c[2]]]
    s3 = [c for c in combinations(range(n), 4) if all(U[a, b] for a, b in combinations(c, 2))]
    return {1: s1, 2: s2, 3: s3}

def build_dowker_complex(A: np.ndarray, mode: str = "sink") -> dict[int, list[tuple[int, ...]]]:
    n = A.shape[0]
    M = A.copy()
    np.fill_diagonal(M, 0)
    if mode == "source":
        M = M.T

    simplices = {1: set(), 2: set(), 3: set()}
    for y in range(n):
        witnessed = [x for x in range(n) if M[x, y] > 0]
        for dim in (1, 2, 3):
            if len(witnessed) >= dim + 1:
                for comb in combinations(witnessed, dim + 1):
                    simplices[dim].add(tuple(sorted(comb)))
    return {d: sorted(v) for d, v in simplices.items()}

def build_directed_flag_complex(A: np.ndarray) -> dict[int, list[tuple[int, ...]]]:
    n = A.shape[0]
    M = A.copy()
    np.fill_diagonal(M, 0)
    s1 = [(i, j) for i in range(n) for j in range(n) if i != j and M[i, j] > 0]
    s2 = [
        (i, j, k)
        for i in range(n) for j in range(n) for k in range(n)
        if len({i, j, k}) == 3 and M[i, j] and M[i, k] and M[j, k]
    ]
    s3 = [
        (i, j, k, l)
        for i, j, k in s2 for l in range(n)
        if l not in (i, j, k) and M[i, l] and M[j, l] and M[k, l]
    ]
    return {1: s1, 2: s2, 3: s3}

def compute_directed_flag_betti(n_nodes: int, d_simplices: dict[int, list[tuple[int, ...]]]) -> dict[int, int]:
    s0 = [(i,) for i in range(n_nodes)]
    s1, s2, s3 = d_simplices[1], d_simplices[2], d_simplices[3]

    def bmat(high_s, low_s):
        if not high_s or not low_s:
            return np.zeros((len(low_s), len(high_s)), dtype=float)
        low_idx = {s: i for i, s in enumerate(low_s)}
        mat = np.zeros((len(low_s), len(high_s)), dtype=float)
        for j, sigma in enumerate(high_s):
            for k in range(len(sigma)):
                face = sigma[:k] + sigma[k + 1 :]
                if face in low_idx:
                    mat[low_idx[face], j] += (-1.0) ** k
        return mat

    def rk(m):
        return int(np.linalg.matrix_rank(m, tol=1e-9)) if m.size else 0

    r1, r2, r3 = rk(bmat(s1, s0)), rk(bmat(s2, s1)), rk(bmat(s3, s2))
    return {
        0: len(s0) - r1,
        1: (len(s1) - r1) - r2,
        2: (len(s2) - r2) - r3,
    }

# ------------------------------------------------------------
# 3. GLMY Path Homology (regular=True vs regular=False)
# ------------------------------------------------------------
def compute_glmy_path_homology(A: np.ndarray, regular: bool = True) -> dict[str, int]:
    """
    Exact GLMY path homology for p = 0, 1, 2.
    - regular=True: degenerate faces with consecutive identical vertices are 0 (GLMY 2012).
    - regular=False: non-regular path homology (preserves reciprocal 2-cycles a <-> b).
    """
    n = A.shape[0]
    M = A.copy()
    np.fill_diagonal(M, 0)

    A0 = [(i,) for i in range(n)]
    A1 = [(i, j) for i in range(n) for j in range(n) if i != j and M[i, j] > 0]
    A2 = [(i, j, k) for (i, j) in A1 for k in range(n) if j != k and M[j, k] > 0]
    A3 = [(i, j, k, l) for (i, j, k) in A2 for l in range(n) if k != l and M[k, l] > 0]

    def nullspace(mat, tol=1e-9):
        if mat.shape[1] == 0:
            return np.zeros((0, 0))
        if mat.shape[0] == 0:
            return np.eye(mat.shape[1])
        _, s, vt = np.linalg.svd(mat, full_matrices=True)
        r = int(np.sum(s > tol))
        return vt[r:].T

    def build_boundary_to_all(paths, allowed_prev_set):
        non_allowed = {}
        rows_full = []
        for j, p in enumerate(paths):
            for k in range(len(p)):
                face = p[:k] + p[k + 1 :]
                if regular and any(face[m] == face[m + 1] for m in range(len(face) - 1)):
                    continue  # degenerate face = 0 in regular path complex
                sign = (-1.0) ** k
                if face not in allowed_prev_set and face not in non_allowed:
                    non_allowed[face] = len(non_allowed)
                rows_full.append((face, j, sign))

        B_non = np.zeros((len(non_allowed), len(paths)), dtype=float)
        allowed_prev_list = sorted(allowed_prev_set)
        allowed_idx = {f: i for i, f in enumerate(allowed_prev_list)}
        B_allowed = np.zeros((len(allowed_prev_list), len(paths)), dtype=float)

        for face, j, sign in rows_full:
            if face in non_allowed:
                B_non[non_allowed[face], j] += sign
            if face in allowed_idx:
                B_allowed[allowed_idx[face], j] += sign
        return B_non, B_allowed

    O1_basis = np.eye(len(A1))
    _, D1_full = build_boundary_to_all(A1, set(A0))

    B2_non, B2_all = build_boundary_to_all(A2, set(A1))
    O2_basis = nullspace(B2_non)
    D2_on_O2 = B2_all @ O2_basis if O2_basis.size else np.zeros((len(A1), 0))

    B3_non, B3_all = build_boundary_to_all(A3, set(A2))
    O3_basis = nullspace(B3_non)
    D3_on_O3 = B3_all @ O3_basis if O3_basis.size else np.zeros((len(A2), 0))

    def rk(m):
        return int(np.linalg.matrix_rank(m, tol=1e-9)) if m.size else 0

    r1 = rk(D1_full @ O1_basis)
    r2 = rk(D2_on_O2)
    r3 = rk(D3_on_O3)

    dim_O1 = O1_basis.shape[1]
    dim_O2 = O2_basis.shape[1]

    return {
        "n_A1": len(A1),
        "n_A2": len(A2),
        "dim_Omega2": dim_O2,
        "beta0": len(A0) - r1,
        "beta1": (dim_O1 - r1) - r2,
        "beta2": (dim_O2 - r2) - r3,
    }

# ------------------------------------------------------------
# 4. Run Filtration Comparison
# ------------------------------------------------------------
def main():
    W = pd.read_csv("output/tna_weights.csv", index_col=0)
    n = len(W)
    W_mat = W.values

    thresholds = np.round(np.linspace(0.02, 0.30, 15), 3)
    records = []

    for th in thresholds:
        A = (W_mat >= th).astype(int)

        uc = build_undirected_clique_complex(A)
        uc_b = compute_simplicial_betti(n, uc)

        sink = build_dowker_complex(A, mode="sink")
        sink_b = compute_simplicial_betti(n, sink)
        source = build_dowker_complex(A, mode="source")

        dfl = build_directed_flag_complex(A)
        dfl_b = compute_directed_flag_betti(n, dfl)

        glmy_reg = compute_glmy_path_homology(A, regular=True)
        glmy_nonreg = compute_glmy_path_homology(A, regular=False)

        records.append({
            "threshold": th,
            "directed_edges": int(np.sum(A - np.diag(np.diag(A)))),
            "undir_tri": len(uc[2]),
            "sink_tri": len(sink[2]),
            "source_tri": len(source[2]),
            "dfl_tri": len(dfl[2]),
            "beta1_undir_clique": uc_b[1],
            "beta1_dowker_sink": sink_b[1],
            "beta1_directed_flag": dfl_b[1],
            "beta1_glmy_reg": glmy_reg["beta1"],
            "beta1_glmy_nonreg": glmy_nonreg["beta1"],
            "dim_Omega2_reg": glmy_reg["dim_Omega2"],
            "dim_Omega2_nonreg": glmy_nonreg["dim_Omega2"],
        })

    df = pd.DataFrame(records)
    df.to_csv("output/poc_filtration_comparison.csv", index=False)

    print("=== Directed TDA (incl. Regular vs Non-Regular GLMY) PoC Summary ===")
    print(df[[
        "threshold", "directed_edges", "undir_tri", "sink_tri", "source_tri", "dfl_tri",
        "beta1_undir_clique", "beta1_dowker_sink", "beta1_directed_flag",
        "beta1_glmy_reg", "beta1_glmy_nonreg"
    ]].to_string(index=False))

if __name__ == "__main__":
    main()
