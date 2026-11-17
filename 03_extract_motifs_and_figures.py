# 03_extract_motifs_and_figures.py

from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.0,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def find_simple_directed_cycles(A: np.ndarray, max_len: int = 6) -> list[tuple[int, ...]]:
    n = A.shape[0]
    cycles = set()

    def dfs(start, curr, path):
        if len(path) > max_len:
            return
        for nxt in range(n):
            if A[curr, nxt] == 0:
                continue
            if nxt == start and len(path) >= 2:
                m_idx = min(range(len(path)), key=lambda i: path[i])
                canon = tuple(path[m_idx:] + path[:m_idx])
                cycles.add(canon)
            elif nxt not in path and nxt > start:
                dfs(start, nxt, path + [nxt])

    for s in range(n):
        dfs(s, s, [s])
    return sorted(cycles, key=lambda c: (len(c), c))

def make_fig1_filtration(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))

    ax = axes[0]
    ax.axvspan(0.24, 0.28, color="#E69F00", alpha=0.16,
               label="Symmetrization blind spot ($\\beta_1^{\\mathrm{sym}}=0$, $\\beta_1^{dFl}=1$)")
    ax.plot(df["threshold"], df["beta1_undir_clique"], "k--o", lw=1.7, ms=4.5, label="Max-symmetrized Clique")
    ax.plot(df["threshold"], df["beta1_directed_flag"], color="#009E73", marker="^", lw=1.8, ms=5.5, label="Directed Flag ($dFl$)")
    ax.plot(df["threshold"], df["beta1_glmy_reg"], color="#D55E00", marker="D", ls="-.", lw=1.8, ms=4.8, label="GLMY Path Homology")
    ax.plot(df["threshold"], df["beta1_dowker_sink"], color="#0072B2", marker="s", ls=":", lw=1.6, ms=4.5, label="Dowker Sink / Source")

    ax.set_title("(a) 1st Betti Number ($\\beta_1$) Across Thresholds")
    ax.set_xlabel("Transition Probability Threshold ($\\tau$)")
    ax.set_ylabel("1st Betti Number ($\\beta_1$)")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylim(-0.2, 6.2)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right", fontsize=7.6)

    ax = axes[1]
    ax.plot(df["threshold"], df["sink_tri"], color="#0072B2", marker="s", lw=2.0, ms=5, label="Dowker Sink 2-Simplices")
    ax.plot(df["threshold"], df["source_tri"], color="#56B4E9", marker="v", lw=2.0, ms=5.5, label="Dowker Source 2-Simplices")
    ax.plot(df["threshold"], df["undir_tri"], "k--o", lw=1.6, ms=4.5, label="Max-symmetrized Clique 2-Simplices")
    ax.fill_between(df["threshold"], df["source_tri"], df["sink_tri"], color="#0072B2", alpha=0.08)
    ax.annotate("Sink–Source Asymmetry\n(20–21 vs. 0 at $\\tau\\in[0.20,0.28]$)",
                xy=(0.26, 20), xytext=(0.145, 48),
                arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.2),
                fontsize=8.5, color="#005587")

    ax.set_title("(b) Higher-Order Convergence vs. Divergence (2-Simplices)")
    ax.set_xlabel("Transition Probability Threshold ($\\tau$)")
    ax.set_ylabel("Number of 2-Simplices")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    fig.savefig("output/fig1_filtration_asymmetry.png", bbox_inches="tight")
    fig.savefig("output/fig1_filtration_asymmetry.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: output/fig1_filtration_asymmetry.png & .pdf")

def make_fig2_mechanism(W: pd.DataFrame, th: float = 0.26):
    states = list(W.index)
    n = len(states)
    M = (W.values >= th).astype(int)
    np.fill_diagonal(M, 0)
    U = ((M + M.T) > 0).astype(int)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    pos = {i: np.array([np.cos(a), np.sin(a)]) for i, a in enumerate(angles)}

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))

    def draw_nodes(ax, highlight_nodes=None):
        highlight_nodes = highlight_nodes or set()
        for i, label in enumerate(states):
            x, y = pos[i]
            fc = "#FDE725" if i in highlight_nodes else "#F0F4F8"
            ec = "#D55E00" if i in highlight_nodes else "#334155"
            ax.scatter(x, y, s=360, facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=4)
            if x < -0.45:
                lx, ly, ha = 1.24 * x, 1.22 * y, "right"
            elif x > 0.45:
                lx, ly, ha = 1.24 * x, 1.22 * y, "left"
            else:
                lx, ly, ha = 1.28 * x, 1.26 * y, "center"
            ax.text(lx, ly, label, ha=ha, va="center", fontsize=8.5,
                    fontweight="bold" if i in highlight_nodes else "normal", zorder=5)
        ax.set_xlim(-1.95, 1.95)
        ax.set_ylim(-1.60, 1.60)
        ax.set_aspect("equal")
        ax.axis("off")

    ax = axes[0]
    undir_tris = [c for c in combinations(range(n), 3) if U[c[0], c[1]] and U[c[0], c[2]] and U[c[1], c[2]]]
    for a, b, c in undir_tris:
        ax.add_patch(Polygon([pos[a], pos[b], pos[c]], closed=True, facecolor="#94A3B8", alpha=0.28, edgecolor="none", zorder=1))
    for i, j in combinations(range(n), 2):
        if U[i, j]:
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="#475569", lw=1.6, zorder=2)
    draw_nodes(ax)
    ax.set_title(f"(a) Max-symmetrized Clique ($\\tau={th:.2f}$)\n2-Simplices={len(undir_tris)}, $\\beta_1=0$ (reciprocal edge collapsed)")

    ax = axes[1]
    cycles = find_simple_directed_cycles(M, max_len=6)
    cycle_edges = set()
    cycle_nodes = set()
    for c in cycles:
        for idx in range(len(c)):
            u, v = c[idx], c[(idx + 1) % len(c)]
            cycle_edges.add((u, v))
            cycle_nodes.add(u)

    for i in range(n):
        for j in range(n):
            if M[i, j]:
                is_cyc = (i, j) in cycle_edges
                color = "#D55E00" if is_cyc else "#64748B"
                lw = 2.1 if is_cyc else 1.2
                rad = 0.16 if M[j, i] else 0.06
                arr = FancyArrowPatch(
                    pos[i], pos[j],
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-|>,head_length=5,head_width=3",
                    color=color, lw=lw, shrinkA=11, shrinkB=11, zorder=3 if is_cyc else 2
                )
                ax.add_patch(arr)
    draw_nodes(ax, highlight_nodes=cycle_nodes)
    ax.set_title(f"(b) Directed Transitions ($\\tau={th:.2f}$)\n$dFl$ retains $\\beta_1^{{dFl}}=1$; GLMY $\\beta_1^{{\\mathrm{{path}}}}=0$")

    ax = axes[2]
    in_deg = M.sum(axis=0)
    out_deg = M.sum(axis=1)
    y_pos = np.arange(n)
    order = np.argsort(in_deg)
    ax.barh(y_pos - 0.18, in_deg[order], height=0.34, color="#0072B2", label="In-degree (Sink)")
    ax.barh(y_pos + 0.18, out_deg[order], height=0.34, color="#56B4E9", label="Out-degree (Source)")
    ax.axvline(3, color="#D55E00", ls="--", lw=1.1, label="Min degree for 2-simplex (=3)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([states[i] for i in order], fontsize=8.5)
    ax.set_xlabel(f"Degree at $\\tau={th:.2f}$")
    ax.set_title("(c) Sink vs. Source Degree Profile\n(Explains Sink=20 vs. Source=0)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    fig.savefig("output/fig2_mechanism_comparison.png", bbox_inches="tight")
    fig.savefig("output/fig2_mechanism_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: output/fig2_mechanism_comparison.png & .pdf")

def main():
    W = pd.read_csv("output/tna_weights.csv", index_col=0)
    df = pd.read_csv("output/poc_filtration_comparison.csv")
    make_fig1_filtration(df)
    make_fig2_mechanism(W, th=0.26)

if __name__ == "__main__":
    main()
