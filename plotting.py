#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

output_dir = "ablation_results"


# ============================================================
# Utility functions
# ============================================================

def summarize_results(eval_returns):
    """
    eval_returns shape: (num_runs, n_evals)
    """
    mean_curve = np.mean(eval_returns, axis=0)
    std_curve = np.std(eval_returns, axis=0)

    summary = {
        "mean_final_return": float(mean_curve[-1]),
        "std_final_return": float(std_curve[-1]),
        "mean_return_over_last_3_evals": float(np.mean(mean_curve[-3:])),
        "best_mean_return": float(np.max(mean_curve)),
    }
    return mean_curve, std_curve, summary


def plot_with_std(ax, x, mean, std, label):
    ax.plot(x, mean, label=label)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)


def config_to_label(config):
    if config["study"] == "exploration":
        if config["policy"] == "egreedy":
            return f"egreedy, eps={config['epsilon']}"
        elif config["policy"] == "softmax":
            return f"softmax, temp={config['temp']}"
    elif config["study"] == "network":
        return f"hidden={config['hidden_dim']}, lr={config['learning_rate']}"
    return str(config)


def parse_config_from_filename(filename):
    """
    Tries to reconstruct the config from the saved filename.
    Handles names like:
      exploration_egreedy_eps_0p01pnpz(RL_env) user@linux-laptop:~/Documents/Leiden_University/Courses/Reinforcement_learning/assignment_2$ /home/user/anaconda3/envs/RL_env/bin/python /home/user/Documents/Leiden_University/Courses/Reinforcement_learning/assignment_2/plotting.py
Loaded summary.json
Loaded 6 exploration result files.
Loaded 6 network result files.
ICE default IO error handler doing an exit(), pid = 323333, errno = 2p
      exploration_softmax_temp_1p0pnpz
      network_hidden_32_lr_0p001pnpz
    or normal .npz names.
    """
    name = os.path.basename(filename)

    if name.startswith("exploration_egreedy_eps_"):
        raw = name[len("exploration_egreedy_eps_"):]
        raw = raw.replace(".npz", "")
        raw = raw.replace("pnpz", "")
        epsilon = float(raw.replace("p", "."))
        return {
            "study": "exploration",
            "policy": "egreedy",
            "epsilon": epsilon,
            "temp": None,
            "hidden_dim": None,
            "learning_rate": None,
        }

    if name.startswith("exploration_softmax_temp_"):
        raw = name[len("exploration_softmax_temp_"):]
        raw = raw.replace(".npz", "")
        raw = raw.replace("pnpz", "")
        temp = float(raw.replace("p", "."))
        return {
            "study": "exploration",
            "policy": "softmax",
            "epsilon": None,
            "temp": temp,
            "hidden_dim": None,
            "learning_rate": None,
        }

    if name.startswith("network_hidden_"):
        raw = name[len("network_hidden_"):]
        raw = raw.replace(".npz", "")
        raw = raw.replace("pnpz", "")

        match = re.match(r"(\d+)_lr_(.+)", raw)
        if match is None:
            raise ValueError(f"Could not parse network filename: {name}")

        hidden_dim = int(match.group(1))
        learning_rate = float(match.group(2).replace("p", "."))

        return {
            "study": "network",
            "policy": None,
            "epsilon": None,
            "temp": None,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
        }

    raise ValueError(f"Unrecognized filename format: {name}")


def load_results_from_npz_files(output_dir):
    exploration_results = []
    network_results = []

    filenames = sorted(os.listdir(output_dir))

    for filename in filenames:
        full_path = os.path.join(output_dir, filename)

        if not os.path.isfile(full_path):
            continue

        if not filename.startswith(("exploration_", "network_")):
            continue

        try:
            data = np.load(full_path)
        except Exception:
            continue

        if "eval_returns" not in data or "eval_timesteps" not in data:
            continue

        config = parse_config_from_filename(filename)
        eval_returns = data["eval_returns"]
        eval_timesteps = data["eval_timesteps"]
        mean_curve, std_curve, summary = summarize_results(eval_returns)

        result = {
            "config": config,
            "eval_timesteps": eval_timesteps,
            "eval_returns": eval_returns,
            "mean_curve": mean_curve,
            "std_curve": std_curve,
            "summary": summary,
        }

        if config["study"] == "exploration":
            exploration_results.append(result)
        elif config["study"] == "network":
            network_results.append(result)

    return exploration_results, network_results


def try_load_summary_json(output_dir):
    summary_path = os.path.join(output_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)
    return None


# ============================================================
# Plotting
# ============================================================

def plot_exploration_results(exploration_results, output_dir):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # sort for nicer legend order
    def sort_key(result):
        cfg = result["config"]
        if cfg["policy"] == "egreedy":
            return (0, cfg["epsilon"])
        return (1, cfg["temp"])

    for result in sorted(exploration_results, key=sort_key):
        label = config_to_label(result["config"])
        plot_with_std(
            ax,
            result["eval_timesteps"],
            result["mean_curve"],
            result["std_curve"],
            label
        )

    plt.xlabel("Environment timesteps")
    plt.ylabel("Mean evaluation return")
    plt.title("Exploration hyperparameter ablation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "exploration_ablation_reloaded.png"),
        dpi=300
    )
    plt.close()


def plot_network_results(network_results, output_dir):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # sort for nicer legend order
    def sort_key(result):
        cfg = result["config"]
        return (cfg["hidden_dim"], cfg["learning_rate"])

    for result in sorted(network_results, key=sort_key):
        label = config_to_label(result["config"])
        plot_with_std(
            ax,
            result["eval_timesteps"],
            result["mean_curve"],
            result["std_curve"],
            label
        )

    plt.xlabel("Environment timesteps")
    plt.ylabel("Mean evaluation return")
    plt.title("Neural network hyperparameter ablation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "network_ablation_reloaded.png"),
        dpi=300
    )
    plt.close()


# ============================================================
# Ranking
# ============================================================

def print_rankings(results, title):
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")

    sorted_results = sorted(
        results,
        key=lambda x: x["summary"]["mean_return_over_last_3_evals"],
        reverse=True
    )

    for i, result in enumerate(sorted_results, start=1):
        label = config_to_label(result["config"])
        score = result["summary"]["mean_return_over_last_3_evals"]
        final_score = result["summary"]["mean_final_return"]
        print(
            f"{i:2d}. {label:<35} "
            f"last3_mean={score:8.3f}, final_mean={final_score:8.3f}"
        )


# ============================================================
# Main
# ============================================================

def main():
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Directory '{output_dir}' does not exist.")

    summary = try_load_summary_json(output_dir)
    if summary is not None:
        print("Loaded summary.json")
    else:
        print("No summary.json found, continuing with only .npz files.")

    exploration_results, network_results = load_results_from_npz_files(output_dir)

    if len(exploration_results) == 0 and len(network_results) == 0:
        raise RuntimeError(
            f"No valid saved result files found in '{output_dir}'."
        )

    print(f"Loaded {len(exploration_results)} exploration result files.")
    print(f"Loaded {len(network_results)} network result files.")

    if exploration_results:
        plot_exploration_results(exploration_results, output_dir)
        print_rankings(exploration_results, "Exploration ablation ranking")

    if network_results:
        plot_network_results(network_results, output_dir)
        print_rankings(network_results, "Network ablation ranking")


if __name__ == "__main__":
    main()