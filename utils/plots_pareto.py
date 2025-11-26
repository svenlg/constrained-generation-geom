import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorsys
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator


# >>>>>>>>>>>>>>>> CONFIGURATION <<<<<<<<<<<<<<<<
PRE_COLOR = "#00AF54"
AM_COLOR  = "#EBE013"   # AM -> yellow
CFO_COLOR  = "#8332AC"  # CFO -> purple
TRUE_COLOR = "#0072B5"  # True -> blue
PREDICTED_COLOR = "#DC9326"  # Predicted -> orange
BOUND_COLOR = "#D62728"  # red for constraint bound

# Dark green for AM-C baselines
BASELINE_COLOR = "#006400"  # dark green
# >>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<

np.random.seed(0)


def style_for_name(name):
    """
    Return (color, label) based on a run/sweep 'value_str'.
    - If it contains 'CFO'  -> purple, label 'CFO'
    - If it contains 'AM'   -> yellow, label 'AM'
    - If it contains 'PRE'  -> green,  label 'PRE'
    Otherwise: (random color, original_name)
    """
    upper = str(name).upper()
    if "CFO" in upper:
        return CFO_COLOR, "CFO"
    elif "AM" in upper:
        return AM_COLOR, "AM"
    elif "PRE" in upper:
        return PRE_COLOR, "PRE"
    color_hex = mcolors.to_hex(np.random.rand(3,))
    return color_hex, str(name)


def collect_runs_for_sweep(sweep_path):
    """
    Returns a dict:
        value_str -> list of paths to run directories for different seeds.
    Expects folders named like '8_0', '8_1', '32_0', '32_1', ...
    """
    groups = {}
    for name in os.listdir(sweep_path):
        run_dir = os.path.join(sweep_path, name)
        if not os.path.isdir(run_dir):
            continue
        if "_" in name:
            value_str, seed_str = name.rsplit("_", 1)
            groups.setdefault(value_str, []).append(run_dir)
        if name.isdigit():
            value_str = name
            groups.setdefault("CFO", []).append(run_dir)
    return groups


def generate_baseline_colors(n, base_hex=BASELINE_COLOR):
    """
    Generate n clearly distinguishable green-ish colors
    with monotonically decreasing brightness.

    - Smallest baseline value -> brightest green
    - Largest baseline value -> darkest green
    """

    if n <= 0:
        return []

    base_rgb = mcolors.to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(*base_rgb)

    # Nonlinear lightness spacing: brighter values spaced further apart
    # Ensures AM 0.01 and AM 0.1 are clearly distinct.
    lightnesses = np.linspace(0.8, 0.0, n) ** 1.7
    lightnesses = 0.9 * lightnesses + 0.20
    colors = []
    for L in lightnesses:
        L = float(np.clip(L, 0.20, 0.85))
        r, g, b = colorsys.hls_to_rgb(h, L, s)
        colors.append(mcolors.to_hex((r, g, b)))

    return colors


def load_metric_arrays(run_dirs, metric):
    """
    Load a given metric ('reward', 'constraint', etc.) from each run_dir.
    Returns a stacked array of shape (n_seeds, T).
    """
    series_list = []
    for rd in run_dirs:
        csv_path = os.path.join(rd, "full_stats.csv")
        if not os.path.isfile(csv_path):
            print(f"Warning: {csv_path} not found, skipping.")
            continue
        df = pd.read_csv(csv_path)
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in {csv_path}")
        series_list.append(df[metric].to_numpy())

    if not series_list:
        return None

    min_len = min(len(s) for s in series_list)
    series_list = [s[:min_len] for s in series_list]
    arr = np.stack(series_list, axis=0)  # (n_seeds, T)
    return arr


def mean_ci_scalar(samples, alpha=0.05):
    """
    Compute mean and (1-alpha)% CI half-width for a 1D array of samples.
    Returns (mean, half_width).
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    mean = samples.mean()
    if n > 1:
        std = samples.std(ddof=1)
        z = 1.96  # for 95% CI
        half_width = z * std / np.sqrt(n)
    else:
        half_width = 0.0
    return mean, half_width


def plot_pareto_cfo_vs_baseline(
    cfo_dir,
    baseline_dir,
    logging_interval=3,   # unused now, but kept for CLI compatibility
    k_update_steps=10,    # unused now, but kept for CLI compatibility
    font_scale=2.0,
    bound=None,
    show_front=False,
    safe_fig=False,
    reward_name="Dipole (in D)",
    constraint_name="Energy (in Ha)",
    out_name="pareto_cfo_vs_baseline",
    baseline_name="AM-C",
    flip_constraint_axis=False,
):
    """
    Pareto plot: x-axis = constraint (Energy), y-axis = reward (Dipole).

    - Baseline_dir: subfolders like '0.01_0', '0.01_1', grouped by value via collect_runs_for_sweep.
      We use final time step mean and 95% CI across seeds for reward & constraint.
      Labelled as '<baseline_name> value', e.g. 'AM-C 0.01'.
    - cfo_dir: each subfolder is a seed for CFO.
      We use ONLY the FINAL time step (one point) and show mean + 95% CI.

    CI is drawn as error bars in both x and y directions.

    Options:
    - bound: if not None, draws a vertical red line at x = bound.
    - show_front: if True, draw Pareto front over all points
      (minimize constraint, maximize reward).
    - flip_constraint_axis: if True, invert the x-axis so that lower energy
      (more negative, i.e. better) appears to the RIGHT. This can make
      'top-right is better' visually intuitive even though the underlying
      optimization is (min constraint, max reward).
    - safe_fig: if True, save to PDF/JPG instead of showing.
    """

    base_fs = 10 * font_scale
    rc = {
        "font.size": base_fs,
        "axes.titlesize": base_fs * 1.2,
        "axes.labelsize": base_fs * 1.1,
        "xtick.labelsize": base_fs * 1.0,
        "ytick.labelsize": base_fs * 1.0,
        "legend.fontsize": base_fs * 0.85,
        "axes.linewidth": 1.5,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # ---------- BASELINES ----------
        baseline_groups = collect_runs_for_sweep(baseline_dir)

        # sort keys numerically so smallest value gets lightest color
        baseline_keys = [k for k in baseline_groups.keys() if k not in ("CFO", "AM", "PRE")]
        baseline_keys_sorted = sorted(baseline_keys, key=float)

        baseline_colors = generate_baseline_colors(len(baseline_keys_sorted))
        baseline_color_map = {k: c for k, c in zip(baseline_keys_sorted, baseline_colors)}


        all_x = []
        all_y = []

        for value_str, run_dirs in sorted(
            baseline_groups.items(),
            key=lambda x: float(x[0]) if x[0] not in ("CFO", "AM", "PRE") else float("inf")
        ):
            reward_arr = load_metric_arrays(run_dirs, "reward")
            constraint_arr = load_metric_arrays(run_dirs, "constraint")
            if reward_arr is None or constraint_arr is None:
                continue

            # Final time step arrays: (n_seeds,)
            r_final = reward_arr[:, -1]
            c_final = constraint_arr[:, -1]

            r_mean, r_hw = mean_ci_scalar(r_final)
            c_mean, c_hw = mean_ci_scalar(c_final)

            color = baseline_color_map.get(value_str, BASELINE_COLOR)
            label = f"{baseline_name}={value_str}"

            # Error bars (CI) and point
            ax.errorbar(c_mean, r_mean, xerr=c_hw, yerr=r_hw,
                fmt="none",
                ecolor=color,
                elinewidth=2.5,
                capsize=4,
                alpha=0.9,
                zorder=2,
            )
            ax.scatter(c_mean, r_mean,
                marker="o",
                color=color,
                edgecolor="black",
                s=120,
                zorder=3,
                label=label,
            )

            all_x.append(c_mean)
            all_y.append(r_mean)

        # ---------- CFO (FINAL ITERATE ONLY) ----------
        cfo_run_dirs = [
            os.path.join(cfo_dir, d)
            for d in os.listdir(cfo_dir)
            if os.path.isdir(os.path.join(cfo_dir, d))
        ]

        if cfo_run_dirs:
            reward_arr = load_metric_arrays(cfo_run_dirs, "reward")
            constraint_arr = load_metric_arrays(cfo_run_dirs, "constraint")

            if reward_arr is not None and constraint_arr is not None:
                # Final time step
                r_final = reward_arr[:, -1]
                c_final = constraint_arr[:, -1]

                r_mean, r_hw = mean_ci_scalar(r_final)
                c_mean, c_hw = mean_ci_scalar(c_final)

                # Error bars (CI) and point for CFO
                ax.errorbar(c_mean, r_mean, xerr=c_hw, yerr=r_hw,
                    fmt="none",
                    ecolor=CFO_COLOR,
                    elinewidth=2.5,
                    capsize=5,
                    alpha=0.95,
                    zorder=4,
                )
                ax.scatter(c_mean, r_mean,
                    marker="s",
                    color=CFO_COLOR,
                    edgecolor="black",
                    s=150,
                    zorder=5,
                    label="CFO",
                )

                all_x.append(c_mean)
                all_y.append(r_mean)

        # ---------- BOUND ----------
        if bound is not None:
            ax.axvline(
                bound,
                linestyle="--",
                linewidth=3.5,
                color=BOUND_COLOR,
                label=f"Bound",
                zorder=1,
            )

        # ---------- PARETO FRONT (in true objective space) ----------
        if show_front and all_x and all_y:
            pts = np.array(list(zip(all_x, all_y)))  # (N, 2)
            # Objective: minimize x (constraint), maximize y (reward)
            order = np.argsort(pts[:, 0])  # sort by constraint
            pts_sorted = pts[order]

            front = []
            best_y = -np.inf
            for x, y in pts_sorted:
                if y > best_y:
                    front.append((x, y))
                    best_y = y

            if len(front) >= 2:
                front = np.array(front)
                ax.plot(
                    front[:, 0],
                    front[:, 1],
                    linestyle="-.",
                    linewidth=3.5,
                    color="black",
                    alpha=0.8,
                    label="Pareto front",
                    zorder=6,
                )

        # ---------- LABELS / FORMAT ----------
        ax.set_xlabel(constraint_name)
        ax.set_ylabel(reward_name)
        ax.set_title("CFO and Baseline")

        # Flip x-axis if requested: visually, better (lower energy) to the right.
        if flip_constraint_axis:
            ax.invert_xaxis()

        ax.grid(True, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        # de-duplicate legend entries
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="lower right")

        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

        fig.tight_layout()

        if safe_fig:
            base_dir = os.path.dirname(baseline_dir)
            out_base = os.path.join(base_dir, "zz_figures")
            os.makedirs(out_base, exist_ok=True)
            out_path = os.path.join(out_base, out_name)
            fig.savefig(f"{out_path}.jpg", bbox_inches="tight")
            fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"Saved Pareto figure to {out_path}.jpg/.pdf")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Pareto plot of CFO vs baseline (constraint vs reward) with 95% CI."
    )
    parser.add_argument("--cfo_dir", type=str, required=True,
                        help="Folder with CFO runs (each seed as a subfolder).")
    parser.add_argument("--baseline_dir", type=str, required=True,
                        help="Folder with baseline runs; subfolders like '0.01_0', '0.01_1', etc.")
    parser.add_argument("--logging_interval", type=int, default=3,
                        help="Env steps between two logged rows in full_stats.csv (kept for compatibility).")
    parser.add_argument("--k_update_steps", type=int, default=10,
                        help="Env-step distance between two k-updates (kept for compatibility).")
    parser.add_argument("--font_scale", type=float, default=2.0,
                        help="Font size scaling factor.")
    parser.add_argument("--bound", type=float, default=None,
                        help="Constraint bound for vertical red line.")
    parser.add_argument("--show_front", action="store_true",
                        help="If set, draw Pareto front over all points.")
    parser.add_argument("--safe_fig", action="store_true",
                        help="Save figures instead of showing them.")
    parser.add_argument("--reward", type=str, default="Dipole (in D)",
                        help="Reward axis label.")
    parser.add_argument("--constraint", type=str, default="Energy (in Ha)",
                        help="Constraint axis label.")
    parser.add_argument("--out_name", type=str, default="pareto_dipole_energy",
                        help="Base name for saved figure.")
    # parser.add_argument("--baseline_name", type=str, default=r"$\mathrm{AM}-\mu$",
    parser.add_argument("--baseline_name", type=str, default=r"$\mu$",
                        help="Name used for baseline in the legend (e.g. 'AM-C').")
    parser.add_argument("--flip_constraint_axis", action="store_false",
                        help="Invert x-axis so lower energy appears to the right.")
    args = parser.parse_args()

    plot_pareto_cfo_vs_baseline(
        cfo_dir=args.cfo_dir,
        baseline_dir=args.baseline_dir,
        logging_interval=args.logging_interval,
        k_update_steps=args.k_update_steps,
        font_scale=args.font_scale,
        bound=args.bound,
        show_front=args.show_front,
        safe_fig=args.safe_fig,
        reward_name=args.reward,
        constraint_name=args.constraint,
        out_name=args.out_name,
        baseline_name=args.baseline_name,
        flip_constraint_axis=args.flip_constraint_axis,
    )


if __name__ == "__main__":
    main()

"""
python utils/plots_pareto.py \
  --cfo_dir /Users/svlg/MasterThesis/v03_geom/aa_experiments/al_dipole_energy \
  --baseline_dir /Users/svlg/MasterThesis/v03_geom/aa_experiments/fixed_baseline_dipole_energy \
  --font_scale 2.8 \
  --bound -80 \
  --safe_fig
"""