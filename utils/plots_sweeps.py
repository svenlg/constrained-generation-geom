import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# >>>>>>>>>>>>>>>> CONFIGURATION <<<<<<<<<<<<<<<<
# Path to the directory that contains the 11_* folders
BASE_DIR = "/cluster/home/sgutjahr/MasterThesis/constrained-generation-geom/aa_experiments"

# Which sweeps to plot.
SWEEPS_TO_PLOT = "am_total_steps_10seeds"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def find_sweep_dirs(base_dir, sweeps_to_plot):
    all_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if sweeps_to_plot is not None:
        all_dirs = [d for d in all_dirs if d in sweeps_to_plot]
    return sorted(all_dirs)


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
        if "_" not in name:
            continue
        value_str, seed_str = name.rsplit("_", 1)
        groups.setdefault(value_str, []).append(run_dir)
    return groups


def load_metric_arrays(run_dirs, metric):
    """
    Load a given metric ('reward', 'constraint', etc.) from each run_dir.
    Returns a stacked array of shape (n_seeds, T).
    Assumes all runs for this value have the same length.
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

    # Optionally trim to shortest length to be safe:
    min_len = min(len(s) for s in series_list)
    series_list = [s[:min_len] for s in series_list]

    arr = np.stack(series_list, axis=0)  # (n_seeds, T)
    return arr


def mean_and_ci(arr, alpha=0.05):
    """
    Compute mean and two-sided (1-alpha)*100% CI across axis 0.
    Uses normal approximation: mean ± z * std / sqrt(n)
    """
    n = arr.shape[0]
    mean = arr.mean(axis=0)
    if n > 1:
        std = arr.std(axis=0, ddof=1)
        z = 1.96  # for 95% CI
        half_width = z * std / np.sqrt(n)
    else:
        half_width = np.zeros_like(mean)
    lower = mean - half_width
    upper = mean + half_width
    return mean, lower, upper

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


def plot_sweep(base_dir, sweep_name, parameter,
               reward="Dipole",
               constraint="Energy",
               font_scale=2.0,
               logging_interval=3,
               safe_fig=False):
    """
    Plot reward + constraint curves with 95% CI using large fonts and scaled x-axis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    sweep_path = os.path.join(base_dir, sweep_name)
    value_to_runs = collect_runs_for_sweep(sweep_path)

    if not value_to_runs:
        print(f"No runs found in {sweep_path}")
        return

    # ---------- font scaling for paper-quality ----------
    base_fs = 10 * font_scale
    rc = {
        "font.size": base_fs,
        "axes.titlesize": base_fs * 1.2,
        "axes.labelsize": base_fs * 1.1,
        "xtick.labelsize": base_fs * 1.0,
        "ytick.labelsize": base_fs * 1.0,
        "legend.fontsize": base_fs * 0.95,
        "axes.linewidth": 1.5,
    }

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False)
        ax_reward, ax_constraint = axes

        xmax = 0

        for value_str, run_dirs in sorted(value_to_runs.items(),
                                          key=lambda x: float(x[0])):

            reward_arr = load_metric_arrays(run_dirs, "reward")
            constraint_arr = load_metric_arrays(run_dirs, "constraint")
            if reward_arr is None or constraint_arr is None:
                continue

            r_mean, r_low, r_high = mean_and_ci(reward_arr)
            c_mean, c_low, c_high = mean_and_ci(constraint_arr)

            # ------ NEW: scale x-axis to real environment steps -------
            steps = np.arange(len(r_mean))
            x = steps * logging_interval
            xmax = max(xmax, x[-1])

            # label = f"{parameter}={value_str}"
            label = f"{value_str}"

            # reward
            ax_reward.plot(x, r_mean, label=label)
            ax_reward.fill_between(x, r_low, r_high, alpha=0.2)

            # constraint
            ax_constraint.plot(x, c_mean, label=label)
            ax_constraint.fill_between(x, c_low, c_high, alpha=0.2)

        # ----- formatting both axes -----
        for ax, title, ylabel in [
            (ax_reward, f"{reward}", f"{reward}"),
            (ax_constraint, f"{constraint}", f"{constraint}")
        ]:
            ax.set_xlabel("Steps")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

            ax.set_xlim(0, xmax)

            ax.legend()

        fig.tight_layout()

        if safe_fig:
            out_path = os.path.join(base_dir, f"{sweep_name}_reward_constraint")
            fig.savefig(f"{out_path}.jpg", bbox_inches="tight")
            fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"Saved figure to {out_path}")


def summarize_sweep_markdown(base_dir, sweep_name, parameter_name):
    """
    Build a markdown table for one sweep.
    Each row: parameter value, reward at first & last step, constraint at first & last step,
    with mean ± 95% CI across seeds.

    Returns a markdown string.
    """
    sweep_path = os.path.join(base_dir, sweep_name)
    value_to_runs = collect_runs_for_sweep(sweep_path)

    if not value_to_runs:
        print(f"No runs found in {sweep_path}")
        return ""

    rows = []
    for value_str, run_dirs in sorted(value_to_runs.items(),
                                      key=lambda x: float(x[0])):

        reward_arr = load_metric_arrays(run_dirs, "reward")
        constraint_arr = load_metric_arrays(run_dirs, "constraint")
        if reward_arr is None or constraint_arr is None:
            continue

        # reward_arr, constraint_arr: shape (n_seeds, T)
        # first and last indices
        r0_mean, r0_hw = mean_ci_scalar(reward_arr[:, 0])
        rT_mean, rT_hw = mean_ci_scalar(reward_arr[:, -1])

        c0_mean, c0_hw = mean_ci_scalar(constraint_arr[:, 0])
        cT_mean, cT_hw = mean_ci_scalar(constraint_arr[:, -1])

        def fmt(m, hw):
            return f"{m:.3f} ± {hw:.3f}"

        rows.append({
            "value": value_str,
            "reward_0": fmt(r0_mean, r0_hw),
            "reward_T": fmt(rT_mean, rT_hw),
            "constraint_0": fmt(c0_mean, c0_hw),
            "constraint_T": fmt(cT_mean, cT_hw),
        })

    # Build markdown
    header = (
        f"| {parameter_name} | reward₀ | rewardₜ | constraint₀ | constraintₜ |\n"
        f"|---|---|---|---|---|\n"
    )
    body_lines = []
    for row in rows:
        line = (
            f"| {row['value']} | "
            f"{row['reward_0']} | {row['reward_T']} | "
            f"{row['constraint_0']} | {row['constraint_T']} |"
        )
        body_lines.append(line)

    table_md = header + "\n".join(body_lines)
    return table_md


def main():
    sweep_dirs = find_sweep_dirs(BASE_DIR, SWEEPS_TO_PLOT)
    if not sweep_dirs:
        print("No 11_* sweeps found. Check BASE_DIR / SWEEPS_TO_PLOT.")
        return

    for sweep_name in sweep_dirs:
        parts = sweep_name.split("_")
        parameter = "_".join(parts[3:])
        print(f"Processing sweep: {parameter}")

        # plots
        plot_sweep(BASE_DIR, sweep_name, parameter, safe_fig=True)

        # table
        table_md = summarize_sweep_markdown(BASE_DIR, sweep_name, parameter)
        print()
        print(f"### Sweep: {parameter}")
        print(table_md)
        print()


if __name__ == "__main__":
    main()
