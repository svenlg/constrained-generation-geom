import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# >>>>>>>>>>>>>>>> CONFIGURATION <<<<<<<<<<<<<<<<
PRE_COLOR = "#00AF54"
AM_COLOR  = "#EBE013"   # AM -> yellow
PURPLE20  = "#8332AC"   # CFO -> purple
TRUE_COLOR = "#0072B5"  # True -> blue
PREDICTED_COLOR = "#DC9326"  # Predicted -> orange
BOUND_COLOR = "#D62728"  # red for constraint bound


def style_for_name(name):
    """
    Return (color, label) based on a run/sweep 'value_str'.
    - If it contains 'CFO'  -> purple, label 'CFO'
    - If it contains 'AM'   -> yellow, label 'AM'
    - If it contains 'PRE'  -> green,  label 'PRE'
    Otherwise: (None, original_name)
    """
    upper = str(name).upper()
    if "CFO" in upper:
        return PURPLE20, "CFO"
    if "AM" in upper:
        return AM_COLOR, "AM"
    if "PRE" in upper:
        return PRE_COLOR, "PRE"
    return None, str(name)


def find_sweep_dirs(base_dir, sweeps_to_plot):
    """
    Filter the directories in base_dir to those listed in sweeps_to_plot.
    sweeps_to_plot should be a list of sweep folder names.
    """
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
        if "_" in name:
            value_str, seed_str = name.rsplit("_", 1)
            groups.setdefault(value_str, []).append(run_dir)
        # just number    
        if name.isdigit():
            value_str = name
            groups.setdefault("CFO", []).append(run_dir)
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


def mean_and_ci(arr):
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


def mean_ci_scalar(samples):
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


def plot_sweep(
    base_dir,
    sweep_name,
    parameter,
    reward="Dipole",
    constraint="Energy",
    font_scale=2.0,
    logging_interval=3,
    k_update_steps=None,
    use_k_axis=False,
    plot_pre=False,
    safe_fig=False,
    am_reward_baseline=None,
    am_constraint_baseline=None,
    bound=None,
):
    """
    Plot reward + constraint curves with 95% CI into two separate figures.

    logging_interval : env steps between two logged rows in full_stats.csv
    k_update_steps   : env-step distance between two k-updates (vertical bars)
    use_k_axis       : if True, x axis is k = env_steps / k_update_steps.
                       If False, x axis is env steps.
    bound            : if not None, draw horizontal red line at this constraint value
                       on the constraint figure only.
    """
    sweep_path = os.path.join(base_dir, sweep_name)
    value_to_runs = collect_runs_for_sweep(sweep_path)

    if not value_to_runs:
        print(f"No runs found in {sweep_path}")
        return

    # ---------- font scaling ----------
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
        # Separate figures for reward and constraint
        fig_reward, ax_reward = plt.subplots(1, 1, figsize=(7, 5))
        fig_constraint, ax_constraint = plt.subplots(1, 1, figsize=(7, 5))

        xmax = 0.0        # max x in plot coordinates
        max_step = 0.0    # max env step (for computing bars later)

        # We'll store CFO curve for later to place dots
        cfo_x = None
        cfo_r_mean = None
        cfo_c_mean = None

        # Sort so "CFO" comes first if present
        if "CFO" in value_to_runs:
            sorted_items = sorted(
                value_to_runs.items(),
                key=lambda x: (x[0] != "CFO", x[0])
            )
        else:
            sorted_items = sorted(
                value_to_runs.items(),
                key=lambda x: float(x[0])
            )

        # We'll keep the last r_mean/c_mean around for PRE baseline
        r_mean = c_mean = r_low = r_high = c_low = c_high = None

        for value_str, run_dirs in sorted_items:

            reward_arr = load_metric_arrays(run_dirs, "reward")
            constraint_arr = load_metric_arrays(run_dirs, "constraint")
            if reward_arr is None or constraint_arr is None:
                continue

            r_mean, r_low, r_high = mean_and_ci(reward_arr)
            c_mean, c_low, c_high = mean_and_ci(constraint_arr)

            T = len(r_mean)
            # env steps for each logged row
            steps_env = np.arange(T) * logging_interval
            max_step = max(max_step, steps_env[-1])

            # ---- choose x-axis ----
            if use_k_axis and (k_update_steps is not None):
                # k = step / k_update_steps
                x = steps_env / float(k_update_steps)
            else:
                # plain env steps
                x = steps_env.astype(float)

            xmax = max(xmax, x[-1])

            # reward (CFO-style)
            ax_reward.plot(x, r_mean, label="CFO", color=PURPLE20)
            ax_reward.fill_between(x, r_low, r_high, alpha=0.2, color=PURPLE20)

            # constraint (CFO-style)
            ax_constraint.plot(x, c_mean, label="CFO", color=PURPLE20)
            ax_constraint.fill_between(x, c_low, c_high, alpha=0.2, color=PURPLE20)

            # if this is the CFO group, remember its curve for dots
            if value_str == "CFO":
                cfo_x = x
                cfo_r_mean = r_mean
                cfo_c_mean = c_mean

        # ----- vertical bars at every k-update -----
        vlines_x = None
        if k_update_steps is not None and k_update_steps > 0:
            if use_k_axis:
                # in k-space, bars at k = 1,2,3,... up to max_k
                max_k = max_step / float(k_update_steps)
                k_vals = np.arange(1, np.floor(max_k) + 1, 1.0)
                vlines_x = k_vals
            else:
                # in env-step space, bars at 0, k, 2k, ...
                vlines_x = np.arange(0, max_step + 1e-9, k_update_steps)

            for ax in (ax_reward, ax_constraint):
                for xv in vlines_x:
                    ax.axvline(x=xv, linestyle="--", alpha=0.3)

        # ----- dots where CFO curve meets each vertical bar -----
        if (vlines_x is not None) and (cfo_x is not None):
            v_arr = np.asarray(vlines_x, dtype=float)

            # for each vertical line, find nearest CFO index
            idxs = [int(np.argmin(np.abs(cfo_x - xv))) for xv in v_arr]
            x_pts = cfo_x[idxs]
            y_reward_pts = cfo_r_mean[idxs]
            y_constraint_pts = cfo_c_mean[idxs]

            ax_reward.scatter(
                x_pts,
                y_reward_pts,
                color=PURPLE20,
                edgecolor="black",
                zorder=5,
                label="_nolegend_",  # don't duplicate legend
            )
            ax_constraint.scatter(
                x_pts,
                y_constraint_pts,
                color=PURPLE20,
                edgecolor="black",
                zorder=5,
                label="_nolegend_",
            )

        # ---- custom x-ticks when using k-axis ----
        if use_k_axis and k_update_steps is not None:
            # integer ticks: 1, 2, ..., floor(max_k)
            max_k = max_step / float(k_update_steps)
            xticks = np.arange(0, int(np.floor(max_k)))
            for ax in (ax_reward, ax_constraint):
                ax.set_xticks(xticks)
                ax.set_xticklabels([str(i + 1) for i in xticks])

        # common x-range for horizontal bands/lines
        xs_baseline = np.array([0.0, xmax]) if xmax > 0 else np.array([0.0, 1.0])

        # ----- AM baseline (mean ± std) as horizontal dashed line + dots -----
        if am_reward_baseline is not None:
            am_mean, am_std = am_reward_baseline
            y_low = am_mean - am_std
            y_high = am_mean + am_std

            ax_reward.axhline(
                am_mean,
                linestyle="--",
                linewidth=2.0,
                color=AM_COLOR,
                label="AM",
            )
            ax_reward.fill_between(
                xs_baseline,
                y_low,
                y_high,
                alpha=0.2,
                color=AM_COLOR,
            )

            # scatter AM at each vertical bar (reward)
            # pre_x0 = 0.0
            # ax_constraint.scatter(
            #     [pre_x0],
            #     [am_mean],
            #     color=AM_COLOR,
            #     edgecolor="black",
            #     zorder=5,
            #     label="_nolegend_",
            # )

        if am_constraint_baseline is not None:
            am_mean_c, am_std_c = am_constraint_baseline
            y_low_c = am_mean_c - am_std_c
            y_high_c = am_mean_c + am_std_c

            ax_constraint.axhline(
                am_mean_c,
                linestyle="--",
                linewidth=2.0,
                color=AM_COLOR,
                label="AM",
            )
            ax_constraint.fill_between(
                xs_baseline,
                y_low_c,
                y_high_c,
                alpha=0.2,
                color=AM_COLOR,
            )

            # scatter AM at each vertical bar (constraint)
            # pre_x0 = 0.0
            # ax_constraint.scatter(
            #     [pre_x0],
            #     [am_mean_c],
            #     color=AM_COLOR,
            #     edgecolor="black",
            #     zorder=5,
            #     label="_nolegend_",
            # )

        # ----- PRE (mean ± std) as horizontal dashed line + dot at beginning -----
        if plot_pre and (r_mean is not None) and (c_mean is not None):
            # we use the first step's mean ± CI from the last curve we computed
            pre_r_mean = r_mean[0]
            pre_r_low = r_low[0]
            pre_r_high = r_high[0]

            pre_c_mean = c_mean[0]
            pre_c_low = c_low[0]
            pre_c_high = c_high[0]

            # reward PRE line + band
            ax_reward.axhline(
                pre_r_mean,
                linestyle="--",
                linewidth=2.0,
                color=PRE_COLOR,
                label="PRE",
            )
            ax_reward.fill_between(
                xs_baseline,
                pre_r_low,
                pre_r_high,
                alpha=0.2,
                color=PRE_COLOR,
            )
            # PRE scatter at the very beginning (x = 0)
            # pre_x0 = 0.0
            # ax_reward.scatter(
            #     [pre_x0],
            #     [pre_r_mean],
            #     color=PRE_COLOR,
            #     edgecolor="black",
            #     zorder=6,
            #     label="_nolegend_",
            # )

            # constraint PRE line + band
            ax_constraint.axhline(
                pre_c_mean,
                linestyle="--",
                linewidth=2.0,
                color=PRE_COLOR,
                label="PRE",
            )
            ax_constraint.fill_between(
                xs_baseline,
                pre_c_low,
                pre_c_high,
                alpha=0.2,
                color=PRE_COLOR,
            )
            # PRE scatter at the very beginning (x = 0)
            # ax_constraint.scatter(
            #     [pre_x0],
            #     [pre_c_mean],
            #     color=PRE_COLOR,
            #     edgecolor="black",
            #     zorder=6,
            #     label="_nolegend_",
            # )

        # ----- constraint bound (horizontal red line on constraint only) -----
        if bound is not None:
            ax_constraint.axhline(
                bound,
                linestyle="-",
                linewidth=2.0,
                color=BOUND_COLOR,
                label="bound",
            )

        # ----- formatting and saving -----
        xlabel = "K" if (use_k_axis and k_update_steps is not None) else "N"

        # reward figure formatting
        ax_reward.set_xlabel(xlabel)
        ax_reward.set_ylabel(f"{reward}")
        ax_reward.set_title("Reward")
        ax_reward.grid(True, alpha=0.3)
        ax_reward.set_xlim(0, xmax)
        ax_reward.legend()
        fig_reward.tight_layout()

        # constraint figure formatting
        ax_constraint.set_xlabel(xlabel)
        ax_constraint.set_ylabel(f"{constraint}")
        ax_constraint.set_title("Constraint")
        ax_constraint.grid(True, alpha=0.3)
        ax_constraint.set_xlim(0, xmax)
        ax_constraint.legend(loc="lower left")
        fig_constraint.tight_layout()

        if safe_fig:
            out_base = os.path.join(sweep_path, sweep_name)

            reward_path = f"{out_base}_reward"
            fig_reward.savefig(f"{reward_path}.jpg", bbox_inches="tight")
            fig_reward.savefig(f"{reward_path}.pdf", bbox_inches="tight")

            constraint_path = f"{out_base}_constraint"
            fig_constraint.savefig(f"{constraint_path}.jpg", bbox_inches="tight")
            fig_constraint.savefig(f"{constraint_path}.pdf", bbox_inches="tight")

            plt.close(fig_reward)
            plt.close(fig_constraint)

            print(f"Saved reward figure to {reward_path}")
            print(f"Saved constraint figure to {constraint_path}")



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
    if "CFO" in value_to_runs:
        sorted_items = sorted(value_to_runs.items(),
                                key=lambda x: (x[0] != "CFO", x[0]))
    else:
        sorted_items = sorted(value_to_runs.items(),
                                  key=lambda x: float(x[0]))
            
    for value_str, run_dirs in sorted_items:

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
    BASE_DIR = "/Users/svlg/MasterThesis/v03_geom/aa_experiments"

    parser = argparse.ArgumentParser()
    parser.add_argument("--logging_interval", "-n", type=int, default=2,
                        help="Env steps between two logged rows (your plotting freq N).")
    parser.add_argument("--sweeps_to_plot", "-s", type=str, default="al_final_v1",
                        help="Comma-separated list of sweep folder names to plot.")
    parser.add_argument("--k_update_steps", "-k", type=int, default=10,
                        help="Env-step distance between two k-updates.")
    parser.add_argument("--use_k_axis", action="store_true",
                        help="Use k instead of env steps on the x-axis.")
    parser.add_argument("--reward", type=str, default="Dipole (in D)",
                        help="Reward name for plot title.")
    parser.add_argument("--constraint", type=str, default="Energy (in Ha)",
                        help="Constraint name for plot title.")
    parser.add_argument("--am_reward_mean", type=float, default=None,
                        help="If set, draw horizontal AM reward mean line.")
    parser.add_argument("--am_reward_std", type=float, default=None,
                        help="Std dev around AM reward mean (shaded band).")
    parser.add_argument("--am_constraint_mean", type=float, default=None,
                        help="If set, draw horizontal AM constraint mean line.")
    parser.add_argument("--am_constraint_std", type=float, default=None,
                        help="Std dev around AM constraint mean (shaded band).")
    parser.add_argument("--bound", type=float, default=None,
                        help="Constraint bound for horizontal red line.")
    parser.add_argument("--plot_pre", action="store_true",
                        help="Plot PRE baseline as horizontal line.")
    parser.add_argument("--font_scale", type=float, default=2.0,
                        help="Font size scaling factor.")
    parser.add_argument("--safe_fig", action="store_true",
                        help="Save figures to files instead of showing them.")
    args = parser.parse_args()

    # AM baselines
    am_reward_baseline = None
    if args.am_reward_mean is not None and args.am_reward_std is not None:
        am_reward_baseline = (args.am_reward_mean, args.am_reward_std)

    am_constraint_baseline = None
    if args.am_constraint_mean is not None and args.am_constraint_std is not None:
        am_constraint_baseline = (args.am_constraint_mean, args.am_constraint_std)

    # Parse comma-separated sweeps_to_plot into a list
    sweep_dirs = [s.strip() for s in args.sweeps_to_plot.split(",") if s.strip()]
    sweep_dirs = find_sweep_dirs(BASE_DIR, sweep_dirs)
    if not sweep_dirs:
        print("No sweeps found. Check BASE_DIR / sweeps_to_plot.")
        return

    for sweep_name in sweep_dirs:
        parts = sweep_name.split("_")
        parameter = "_".join(parts[3:]) if len(parts) > 3 else sweep_name
        print(f"Processing sweep: {parameter}")

        plot_sweep(
            BASE_DIR,
            sweep_name,
            parameter,
            reward=args.reward,
            constraint=args.constraint,
            font_scale=args.font_scale,
            logging_interval=args.logging_interval,
            k_update_steps=args.k_update_steps,
            use_k_axis=args.use_k_axis,
            safe_fig=args.safe_fig,
            plot_pre=args.plot_pre,
            am_reward_baseline=am_reward_baseline,
            am_constraint_baseline=am_constraint_baseline,
            bound=args.bound,
        )

        table_md = summarize_sweep_markdown(BASE_DIR, sweep_name, parameter)
        print()
        print(f"### Sweep: {parameter}")
        print(table_md)
        print()


if __name__ == "__main__":
    main()


"""Example usage:
python utils/plots_sweeps_new.py \
  --sweeps_to_plot al_dipole_energy \
  --use_k_axis \
  --am_reward_mean 8.344 \
  --am_reward_std 0.153 \
  --am_constraint_mean -78.011 \
  --am_constraint_std 0.734 \
  --bound -80 \
  --plot_pre \
  --safe_fig
"""
