#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Change this import if your DQN file has a different name
from DQN import DQN_run


# ============================================================
# Settings
# ============================================================

base_output_dir = "experiment_results_target_replay"
experiment_name = "dqn_variant_results_mid_settings_full_rep"
output_dir = os.path.join(base_output_dir, experiment_name)

n_timesteps = 1_000_000
max_episode_length = 500
gamma = 0.99

policy = "egreedy"
epsilon = 0.1
temp = None

learning_rate = 5e-4
hidden_dim = 128
env_steps_per_update = 100
target_update_freq = 200

min_replay_size = 1000

eval_interval = 10_000
n_eval_episodes = 10
num_runs = 5

base_seed = 42


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
    replay = config["use_replay_buffer"]
    target = config["use_target_network"]

    if not replay and not target:
        return "No replay, no target"
    if replay and not target:
        return "Replay buffer only"
    if not replay and target:
        return "Target network only"
    if replay and target:
        return "Replay + target"

    return str(config)


def config_to_filename(config):
    replay_str = "replay" if config["use_replay_buffer"] else "noreplay"
    target_str = "target" if config["use_target_network"] else "notarget"
    return f"variant_{replay_str}_{target_str}.npz"


def run_single_config(config):
    all_eval_returns = []
    eval_timesteps_reference = None

    print(f"\nRunning config: {config_to_label(config)}")

    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        print(f"  Run {run_idx + 1}/{num_runs} with seed={seed}")

        eval_returns, eval_timesteps = DQN_run(
            n_timesteps=n_timesteps,
            max_episode_length=max_episode_length,
            learning_rate=learning_rate,
            gamma=gamma,
            policy=policy,
            epsilon=epsilon,
            temp=temp,
            hidden_dim=hidden_dim,
            env_steps_per_update=env_steps_per_update,
            plot=False,
            eval_interval=eval_interval,
            n_eval_episodes=n_eval_episodes,
            use_replay_buffer=config["use_replay_buffer"],
            min_replay_size=min_replay_size,
            use_target_network=config["use_target_network"],
            target_update_freq=target_update_freq,
            seed=seed,
        )

        all_eval_returns.append(eval_returns)

        if eval_timesteps_reference is None:
            eval_timesteps_reference = eval_timesteps
        else:
            if not np.array_equal(eval_timesteps_reference, eval_timesteps):
                raise RuntimeError("Evaluation timesteps differ across runs.")

    all_eval_returns = np.array(all_eval_returns)
    mean_curve, std_curve, summary = summarize_results(all_eval_returns)

    result = {
        "config": config,
        "eval_timesteps": eval_timesteps_reference,
        "eval_returns": all_eval_returns,
        "mean_curve": mean_curve,
        "std_curve": std_curve,
        "summary": summary,
    }
    return result


def save_result(result, output_dir):
    filename = config_to_filename(result["config"])
    path = os.path.join(output_dir, filename)

    np.savez(
        path,
        eval_returns=result["eval_returns"],
        eval_timesteps=result["eval_timesteps"],
    )


def save_summary_json(results, output_dir):
    summary_dict = {
        "settings": {
            "n_timesteps": n_timesteps,
            "max_episode_length": max_episode_length,
            "gamma": gamma,
            "policy": policy,
            "epsilon": epsilon,
            "temp": temp,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "env_steps_per_update": env_steps_per_update,
            "target_update_freq": target_update_freq,
            "eval_interval": eval_interval,
            "n_eval_episodes": n_eval_episodes,
            "num_runs": num_runs,
            "base_seed": base_seed,
        },
        "results": []
    }

    for result in results:
        summary_dict["results"].append({
            "label": config_to_label(result["config"]),
            "config": result["config"],
            "summary": result["summary"],
            "filename": config_to_filename(result["config"]),
        })

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_dict, f, indent=2)


# ============================================================
# Plotting
# ============================================================

def plot_variant_results(results, output_dir):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    def sort_key(result):
        cfg = result["config"]
        return (cfg["use_replay_buffer"], cfg["use_target_network"])

    for result in sorted(results, key=sort_key):
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
    plt.title("DQN variant comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "dqn_variant_comparison.png"),
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
            f"{i:2d}. {label:<25} "
            f"last3_mean={score:8.3f}, final_mean={final_score:8.3f}"
        )


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(output_dir, exist_ok=True)

    configs = [
        {
            "use_replay_buffer": False,
            "use_target_network": False,
        },
        {
            "use_replay_buffer": True,
            "use_target_network": False,
        },
        {
            "use_replay_buffer": False,
            "use_target_network": True,
        },
        {
            "use_replay_buffer": True,
            "use_target_network": True,
        },
    ]

    results = []

    for config in configs:
        result = run_single_config(config)
        results.append(result)
        save_result(result, output_dir)

    save_summary_json(results, output_dir)
    plot_variant_results(results, output_dir)
    print_rankings(results, "DQN variant ranking")


if __name__ == "__main__":
    main()