#!/usr/bin/env python3
"""
Reads results_naive.csv, results_cube.csv, results_mpi.csv, results_stacked.csv
and produces:

Part I (naive, cube, mpi):
  - Total runtime table (n vs p)  + runtime vs p chart
  - Relative speedup table + chart
  - Parallel efficiency table + chart
  - Step-2-only runtime table (n vs p) + runtime vs p chart
  - Step-2 speedup table + chart
  - Step-2 efficiency table + chart

Part II (stacked):
  - Same set of tables and charts as Part I but for stacked only

All charts saved as PNGs.  Tables printed to stdout and saved to tables.txt.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))

files = {
    "naive":   os.path.join(script_dir, "results_naive.csv"),
    "cube":    os.path.join(script_dir, "results_cube.csv"),
    "mpi":     os.path.join(script_dir, "results_mpi.csv"),
    "stacked": os.path.join(script_dir, "results_stacked.csv"),
}

frames = []
for name, path in files.items():
    df = pd.read_csv(path)
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

# Sorted unique values
all_n = sorted(data["n"].unique())
all_p = sorted(data["p"].unique())

# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────
def make_pivot(df, value_col):
    """Pivot to n-rows x p-columns table."""
    return df.pivot_table(index="n", columns="p", values=value_col, aggfunc="first")


def speedup_table(runtime_tbl):
    """Relative speedup: T(p=1) / T(p)  for each n."""
    t1 = runtime_tbl[1] if 1 in runtime_tbl.columns else runtime_tbl.iloc[:, 0]
    return runtime_tbl.apply(lambda col: t1 / col)


def efficiency_table(sp_tbl):
    """Efficiency: S(p) / p."""
    return sp_tbl.apply(lambda col: col / col.name)


def print_table(title, tbl, fh):
    header = f"\n{'='*60}\n{title}\n{'='*60}"
    print(header)
    print(tbl.to_string())
    fh.write(header + "\n")
    fh.write(tbl.to_string() + "\n\n")


# ──────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────
MARKERS = {"naive": "o", "cube": "s", "mpi": "^", "stacked": "D"}
COLORS  = {"naive": "tab:blue", "cube": "tab:orange", "mpi": "tab:green", "stacked": "tab:red"}


def plot_vs_p(tables_dict, ylabel, title, filename, n_values=None):
    """
    tables_dict: {impl_name: pivot_table}
    Plots value vs p for selected n values, one curve per impl, subplots per n.
    """
    if n_values is None:
        # pick a few representative n values
        candidates = [2**k for k in [10, 14, 17, 20]]
        n_values = [n for n in candidates if all(n in t.index for t in tables_dict.values())]
        if not n_values:
            n_values = list(list(tables_dict.values())[0].index[-3:])

    ncols = min(len(n_values), 3)
    nrows = (len(n_values) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=14, y=1.02)

    for idx, n_val in enumerate(n_values):
        ax = axes[idx // ncols][idx % ncols]
        for impl, tbl in tables_dict.items():
            if n_val not in tbl.index:
                continue
            row = tbl.loc[n_val].dropna()
            ax.plot(row.index, row.values, marker=MARKERS.get(impl, "x"),
                    color=COLORS.get(impl, None), label=impl)
        ax.set_xlabel("p (number of processes)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"n = {n_val}")
        ax.set_xscale("log", base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # hide unused subplots
    for idx in range(len(n_values), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    out = os.path.join(script_dir, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {filename}")


# ──────────────────────────────────────────────
# Part I: naive, cube, mpi
# ──────────────────────────────────────────────
part1_impls = ["naive", "cube", "mpi"]
part1 = data[data["impl"].isin(part1_impls)]

with open(os.path.join(script_dir, "tables.txt"), "w") as fh:

    # --- Total runtime ---
    rt_tables = {}
    sp_tables = {}
    ef_tables = {}
    for impl in part1_impls:
        sub = part1[part1["impl"] == impl]
        rt = make_pivot(sub, "t_total")
        sp = speedup_table(rt)
        ef = efficiency_table(sp)
        rt_tables[impl] = rt
        sp_tables[impl] = sp
        ef_tables[impl] = ef
        print_table(f"Part I – Total Runtime: {impl}", rt, fh)
        print_table(f"Part I – Total Speedup: {impl}", sp, fh)
        print_table(f"Part I – Total Efficiency: {impl}", ef, fh)

    plot_vs_p(rt_tables, "Total Runtime (s)", "Part I – Total Runtime vs p", "part1_total_runtime.png")
    plot_vs_p(sp_tables, "Speedup", "Part I – Total Speedup vs p", "part1_total_speedup.png")
    plot_vs_p(ef_tables, "Efficiency", "Part I – Total Efficiency vs p", "part1_total_efficiency.png")

    # --- Step-2 (all-reduce only) runtime ---
    rt2_tables = {}
    sp2_tables = {}
    ef2_tables = {}
    for impl in part1_impls:
        sub = part1[part1["impl"] == impl]
        rt2 = make_pivot(sub, "t_step2")
        sp2 = speedup_table(rt2)
        ef2 = efficiency_table(sp2)
        rt2_tables[impl] = rt2
        sp2_tables[impl] = sp2
        ef2_tables[impl] = ef2
        print_table(f"Part I – Step2 Runtime: {impl}", rt2, fh)
        print_table(f"Part I – Step2 Speedup: {impl}", sp2, fh)
        print_table(f"Part I – Step2 Efficiency: {impl}", ef2, fh)

    plot_vs_p(rt2_tables, "Step-2 Runtime (s)", "Part I – All-Reduce Runtime vs p", "part1_step2_runtime.png")
    plot_vs_p(sp2_tables, "Speedup", "Part I – All-Reduce Speedup vs p", "part1_step2_speedup.png")
    plot_vs_p(ef2_tables, "Efficiency", "Part I – All-Reduce Efficiency vs p", "part1_step2_efficiency.png")

    # ──────────────────────────────────────────
    # Part II: stacked
    # ──────────────────────────────────────────
    part2 = data[data["impl"] == "stacked"]
    stacked_tables = {}

    # --- Total runtime ---
    rt_s = make_pivot(part2, "t_total")
    sp_s = speedup_table(rt_s)
    ef_s = efficiency_table(sp_s)
    print_table("Part II – Total Runtime: stacked", rt_s, fh)
    print_table("Part II – Total Speedup: stacked", sp_s, fh)
    print_table("Part II – Total Efficiency: stacked", ef_s, fh)

    plot_vs_p({"stacked": rt_s}, "Total Runtime (s)", "Part II – Total Runtime vs p", "part2_total_runtime.png")
    plot_vs_p({"stacked": sp_s}, "Speedup", "Part II – Total Speedup vs p", "part2_total_speedup.png")
    plot_vs_p({"stacked": ef_s}, "Efficiency", "Part II – Total Efficiency vs p", "part2_total_efficiency.png")

    # --- Step-2 runtime ---
    rt2_s = make_pivot(part2, "t_step2")
    sp2_s = speedup_table(rt2_s)
    ef2_s = efficiency_table(sp2_s)
    print_table("Part II – Step2 Runtime: stacked", rt2_s, fh)
    print_table("Part II – Step2 Speedup: stacked", sp2_s, fh)
    print_table("Part II – Step2 Efficiency: stacked", ef2_s, fh)

    plot_vs_p({"stacked": rt2_s}, "Step-2 Runtime (s)", "Part II – Stacked Reduce Runtime vs p", "part2_step2_runtime.png")
    plot_vs_p({"stacked": sp2_s}, "Speedup", "Part II – Stacked Reduce Speedup vs p", "part2_step2_speedup.png")
    plot_vs_p({"stacked": ef2_s}, "Efficiency", "Part II – Stacked Reduce Efficiency vs p", "part2_step2_efficiency.png")

print("\nAll tables written to tables.txt")
print("All charts saved as PNG files.")
