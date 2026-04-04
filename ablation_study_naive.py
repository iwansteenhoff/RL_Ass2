#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import hashlib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DQN import DQN_run

# Settings

# Required ablation hyperparameters (3 values each)
learning_rate_set = [1e-4, 5e-4, 1e-3]
env_steps_per_update_set = [20, 100, 500]      # small, medium, high
hidden_layer_size_set = [32, 128, 256]
epsilon_set = [0.01, 0.1, 0.3]

# fixed training settings
n_timesteps = 1_000_000
max_episode_length = 500
gamma = 0.99
num_runs = 5
eval_interval = 10000
n_eval_episodes = 10

# reproducibility
base_seed = 42

# defaults when not ablating a parameter
default_learning_rate = 5e-4
default_env_steps_per_update = 100
default_hidden_dim = 128
default_policy = "egreedy"
default_epsilon = 0.1

# output directory
base_output_dir = "experiment_results_HPO"
experiment_name = "ablation_results_required_hparams"
output_dir = os.path.join(base_output_dir, experiment_name)

os.makedirs(output_dir, exist_ok=True)


# Utility functions

def safe_float_str(x):
    return str(x).replace(".", "p")


def summarize_results(eval_returns):
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
    study = config["study"]

    if study == "learning_rate":
        return f"lr={config['learning_rate']}"
    elif study == "update_ratio":
        return f"steps/update={config['env_steps_per_update']}"
    elif study == "network_size":
        return f"hidden={config['hidden_dim']}"
    elif study == "exploration":
        return f"epsilon={config['epsilon']}"

    return str(config)


def config_filename(config):
    study = config["study"]

    if study == "learning_rate":
        return f"learning_rate_lr_{safe_float_str(config['learning_rate'])}.npz"
    elif study == "update_ratio":
        return f"update_ratio_spu_{config['env_steps_per_update']}.npz"
    elif study == "network_size":
        return f"network_size_hidden_{config['hidden_dim']}.npz"
    elif study == "exploration":
        return f"exploration_eps_{safe_float_str(config['epsilon'])}.npz"

    raise ValueError(f"Unknown config: {config}")


def config_seed_offset(config):
    """
    Stable integer derived from config contents, so each config gets its own seed block.
    """
    config_str = json.dumps(config, sort_keys=True)
    digest = hashlib.md5(config_str.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def repetition_seed(config, run_idx):
    """
    Deterministic seed for each config/repetition pair.
    """
    return base_seed + config_seed_offset(config) + run_idx


def save_partial_result(filepath, config, eval_returns, eval_timesteps, completed_runs):
    np.savez(
        filepath,
        config_json=json.dumps(config, sort_keys=True),
        eval_returns=np.array(eval_returns, dtype=np.float32),
        eval_timesteps=np.array(eval_timesteps, dtype=np.int64),
        completed_runs=np.array(completed_runs, dtype=np.int64),
    )


def load_partial_result(filepath):
    data = np.load(filepath, allow_pickle=True)

    config = json.loads(str(data["config_json"]))
    eval_returns = data["eval_returns"]
    eval_timesteps = data["eval_timesteps"]
    completed_runs = int(data["completed_runs"])

    return config, eval_returns, eval_timesteps, completed_runs


def load_saved_result_as_result_dict(filepath):
    config, eval_returns, eval_timesteps, completed_runs = load_partial_result(filepath)

    if completed_runs == 0:
        raise ValueError(f"No completed runs stored in {filepath}")

    mean_curve, std_curve, summary = summarize_results(eval_returns)

    return {
        "config": config,
        "eval_timesteps": eval_timesteps,
        "eval_returns": eval_returns,
        "mean_curve": mean_curve,
        "std_curve": std_curve,
        "summary": summary,
        "completed_runs": completed_runs,
    }


# Resume-aware runner

def run_or_resume_config(config):
    filepath = os.path.join(output_dir, config_filename(config))

    if os.path.exists(filepath):
        saved_config, saved_eval_returns, saved_eval_timesteps, completed_runs = load_partial_result(filepath)

        if saved_config != config:
            raise ValueError(
                f"Saved config does not match requested config for file {filepath}."
            )

        print(
            f"\nFound existing file for {config_to_label(config)} "
            f"-> {completed_runs}/{num_runs} repetitions completed"
        )

        all_eval_returns = [saved_eval_returns[i] for i in range(completed_runs)]
        eval_timesteps_ref = saved_eval_timesteps if completed_runs > 0 else None
        start_run = completed_runs
    else:
        print(f"\nStarting new config: {config_to_label(config)}")
        all_eval_returns = []
        eval_timesteps_ref = None
        start_run = 0

    if start_run >= num_runs:
        print(f"Config already complete: {config_to_label(config)}")
        return load_saved_result_as_result_dict(filepath)

    for run_idx in range(start_run, num_runs):
        run_seed = repetition_seed(config, run_idx)
        print(f"  repetition {run_idx + 1}/{num_runs} | seed={run_seed}")

        eval_returns, eval_timesteps = DQN_run(
            n_timesteps=n_timesteps,
            max_episode_length=max_episode_length,
            learning_rate=config["learning_rate"],
            gamma=gamma,
            policy=config["policy"],
            epsilon=config["epsilon"],
            temp=None,
            hidden_dim=config["hidden_dim"],
            env_steps_per_update=config["env_steps_per_update"],
            plot=False,
            eval_interval=eval_interval,
            n_eval_episodes=n_eval_episodes,
            use_replay_buffer=False,
            use_target_network=False,
            target_update_freq=1000,
            seed=run_seed,
        )

        if eval_timesteps_ref is None:
            eval_timesteps_ref = eval_timesteps
        else:
            if len(eval_timesteps_ref) != len(eval_timesteps) or not np.all(eval_timesteps_ref == eval_timesteps):
                raise ValueError("Evaluation timesteps differ across repetitions.")

        all_eval_returns.append(eval_returns)

        completed_runs = len(all_eval_returns)
        save_partial_result(
            filepath=filepath,
            config=config,
            eval_returns=np.array(all_eval_returns),
            eval_timesteps=eval_timesteps_ref,
            completed_runs=completed_runs,
        )

        print(f"  saved progress: {completed_runs}/{num_runs} repetitions")

    return load_saved_result_as_result_dict(filepath)


# Config generation

def get_learning_rate_configs():
    configs = []
    for lr in learning_rate_set:
        configs.append({
            "study": "learning_rate",
            "learning_rate": lr,
            "env_steps_per_update": default_env_steps_per_update,
            "hidden_dim": default_hidden_dim,
            "policy": default_policy,
            "epsilon": default_epsilon,
        })
    return configs


def get_update_ratio_configs():
    configs = []
    for steps_per_update in env_steps_per_update_set:
        configs.append({
            "study": "update_ratio",
            "learning_rate": default_learning_rate,
            "env_steps_per_update": steps_per_update,
            "hidden_dim": default_hidden_dim,
            "policy": default_policy,
            "epsilon": default_epsilon,
        })
    return configs


def get_network_size_configs():
    configs = []
    for hidden_dim in hidden_layer_size_set:
        configs.append({
            "study": "network_size",
            "learning_rate": default_learning_rate,
            "env_steps_per_update": default_env_steps_per_update,
            "hidden_dim": hidden_dim,
            "policy": default_policy,
            "epsilon": default_epsilon,
        })
    return configs


def get_exploration_configs():
    configs = []
    for epsilon in epsilon_set:
        configs.append({
            "study": "exploration",
            "learning_rate": default_learning_rate,
            "env_steps_per_update": default_env_steps_per_update,
            "hidden_dim": default_hidden_dim,
            "policy": default_policy,
            "epsilon": epsilon,
        })
    return configs


# Experiment runners

def run_learning_rate_ablation():
    return [run_or_resume_config(config) for config in get_learning_rate_configs()]


def run_update_ratio_ablation():
    return [run_or_resume_config(config) for config in get_update_ratio_configs()]


def run_network_size_ablation():
    return [run_or_resume_config(config) for config in get_network_size_configs()]


def run_exploration_ablation():
    return [run_or_resume_config(config) for config in get_exploration_configs()]


# Loading saved results

def load_all_saved_results():
    results_by_study = {
        "learning_rate": [],
        "update_ratio": [],
        "network_size": [],
        "exploration": [],
    }

    for filename in sorted(os.listdir(output_dir)):
        if not filename.endswith(".npz"):
            continue

        filepath = os.path.join(output_dir, filename)

        try:
            result = load_saved_result_as_result_dict(filepath)
        except Exception:
            continue

        if result["completed_runs"] < num_runs:
            continue

        study = result["config"]["study"]
        if study in results_by_study:
            results_by_study[study].append(result)

    return results_by_study


# Saving summaries

def save_summary_json(results_by_study):
    summary_dict = {
        "settings": {
            "n_timesteps": n_timesteps,
            "max_episode_length": max_episode_length,
            "gamma": gamma,
            "num_runs": num_runs,
            "eval_interval": eval_interval,
            "n_eval_episodes": n_eval_episodes,
            "base_seed": base_seed,
            "default_learning_rate": default_learning_rate,
            "default_env_steps_per_update": default_env_steps_per_update,
            "default_hidden_dim": default_hidden_dim,
            "default_policy": default_policy,
            "default_epsilon": default_epsilon,
            "backend": "naive.py / DQN_run with vectorized environments",
            "resume_supported": True,
            "reproducible": True,
        },
        "results": {},
    }

    for study, results in results_by_study.items():
        summary_dict["results"][study] = []
        for result in results:
            summary_dict["results"][study].append({
                "config": result["config"],
                "summary": result["summary"],
            })

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary_dict, f, indent=4)


# Plotting

def plot_study(results, title, filename):
    if len(results) == 0:
        return

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for result in results:
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
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


# Rankings

def print_rankings(results, title):
    if len(results) == 0:
        return

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
            f"{i:2d}. {label:<30} "
            f"last3_mean={score:8.3f}, final_mean={final_score:8.3f}"
        )


# Main

def main():
    run_learning_rate_ablation()
    run_update_ratio_ablation()
    run_network_size_ablation()
    run_exploration_ablation()

    results_by_study = load_all_saved_results()

    save_summary_json(results_by_study)

    plot_study(
        results_by_study["learning_rate"],
        "Learning rate ablation",
        "learning_rate_ablation.png"
    )
    plot_study(
        results_by_study["update_ratio"],
        "Update-to-data ratio ablation",
        "update_ratio_ablation.png"
    )
    plot_study(
        results_by_study["network_size"],
        "Network size ablation",
        "network_size_ablation.png"
    )
    plot_study(
        results_by_study["exploration"],
        "Exploration factor ablation",
        "exploration_ablation.png"
    )

    print_rankings(results_by_study["learning_rate"], "Learning rate ranking")
    print_rankings(results_by_study["update_ratio"], "Update-to-data ratio ranking")
    print_rankings(results_by_study["network_size"], "Network size ranking")
    print_rankings(results_by_study["exploration"], "Exploration factor ranking")

    print(f"\nAll complete results saved in: {output_dir}")


if __name__ == "__main__":
    main()