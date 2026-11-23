"""Run online AO on the same episode layouts, reward and metrics as MAPPO."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from algorithm.ppo.main import _build_environment
from algorithm.ppo.file_io import replace_file
from algorithm.traditional_AO.plotting import AOPlotter
from algorithm.traditional_AO.solver import AOConfig, AOSolver
from system_model.params import get_system_params


EPISODE_FIELDS = [
    "episode", "env_steps", "policy_version", "return", "mean_reward", "length",
    "outage_rate", "num_eves", "mean_raw_security_gap", "mean_bob_capacity",
    "mean_max_eve_capacity", "max_bob_capacity", "max_eve_capacity",
    "mean_bob_jammer_inr", "ris_repair_rate",
]
PPO_UPDATE_FIELDS = [
    "update", "total_episodes", "total_env_steps", "mean_episode_return",
    "mean_outage_rate", "satellite_actor_loss", "haps_ris_actor_loss",
    "actor_loss", "critic_loss", "satellite_entropy", "haps_ris_entropy",
    "entropy", "satellite_approx_kl", "haps_ris_approx_kl", "approx_kl",
    "satellite_clip_fraction", "haps_ris_clip_fraction", "clip_fraction",
    "entropy_coef", "satellite_early_stop", "haps_ris_early_stop",
    "satellite_stop_kl", "haps_ris_stop_kl",
]
AO_FIELDS = ["mean_initial_objective", "mean_final_objective", "mean_iterations", "mean_evaluations", "mean_solve_seconds"]
TRACE_FIELDS = ["episode", "step", "iteration", "objective_before", "objective_after", "improvement", "beam_gain", "phase_gain", "mode_gain", "evaluations"]
STEP_FIELDS = ["episode", "step", "reward", "bob_capacity", "max_eve_capacity", "raw_security_gap", "outage", "initial_objective", "objective", "iterations", "evaluations", "solve_seconds", "satellite_power", "ris_power", "requested_active_groups", "executed_active_elements", "ris_action_repaired"]


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    replace_file(temporary, path)


def run_experiment(params, config, episodes, episodes_per_update, run_dir):
    """A batch is only a reporting boundary; AO does not learn between episodes."""
    if episodes <= 0 or episodes_per_update <= 0:
        raise ValueError("episode counts must be positive")
    run_dir = Path(run_dir).resolve()
    package_dir = Path(__file__).resolve().parent
    if not run_dir.is_relative_to(package_dir):
        raise ValueError("AO output must stay inside algorithm/traditional_AO")
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "algorithm": "AO", "information": "oracle_full_csi",
        "objective": "instantaneous minimum secrecy margin",
        "ao_config": asdict(config), "system_params": vars(params),
        "target_episodes": episodes, "episodes_per_update": episodes_per_update,
        "rl_metrics": "Not applicable; corresponding CSV cells are empty",
    }
    write_json(run_dir / "config.json", metadata)
    env, solver = _build_environment(params), AOSolver(config)
    plotter = AOPlotter(run_dir / "training_curves.png")
    total_steps, update = 0, 0
    batch_episodes, batch_steps, all_episodes = [], [], []
    print(f"AO results: {run_dir}", flush=True)
    print("Oracle full-CSI AO; finite-difference projection + greedy group modes.", flush=True)
    try:
        with (
            (run_dir / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as ef,
            (run_dir / "update_metrics.csv").open("w", newline="", encoding="utf-8") as uf,
            (run_dir / "ao_iterations.csv").open("w", newline="", encoding="utf-8") as tf,
            (run_dir / "step_metrics.csv").open("w", newline="", encoding="utf-8") as sf,
        ):
            ew = csv.DictWriter(ef, fieldnames=EPISODE_FIELDS)
            uw = csv.DictWriter(uf, fieldnames=PPO_UPDATE_FIELDS + AO_FIELDS)
            tw = csv.DictWriter(tf, fieldnames=TRACE_FIELDS)
            sw = csv.DictWriter(sf, fieldnames=STEP_FIELDS)
            for writer in (ew, uw, tw, sw):
                writer.writeheader()
            for episode in range(1, episodes + 1):
                env.reset(seed=int(params.seed)) if episode == 1 else env.reset()
                solver.reset()
                rewards, bob, eve, inr, repairs = [], [], [], [], []
                for step in range(1, env.horizon + 1):
                    result = solver.solve(env)
                    _, reward, terminated, truncated, info = env.step(result.action)
                    for entry in result.trace:
                        tw.writerow({"episode": episode, "step": step, **entry})
                    record = {
                        "episode": episode, "step": step, "reward": reward,
                        "bob_capacity": float(np.mean(info["bob_capacity"])),
                        "max_eve_capacity": float(np.mean(info["max_eve_capacity"])),
                        "raw_security_gap": float(np.min(info["raw_security_gap"])),
                        "outage": int(info["outage_sample"]),
                        "initial_objective": result.initial_objective,
                        "objective": result.objective, "iterations": len(result.trace),
                        "evaluations": result.evaluations, "solve_seconds": result.solve_seconds,
                        "satellite_power": info["satellite_power"], "ris_power": info["ris_power"],
                        "requested_active_groups": int(result.groups.mode.sum()),
                        "executed_active_elements": info["ris_active_element_count"],
                        "ris_action_repaired": int(info["ris_action_repaired"]),
                    }
                    sw.writerow(record)
                    batch_steps.append(record)
                    rewards.append(reward)
                    bob.append(float(np.max(info["bob_capacity"])))
                    eve.append(float(np.max(info["max_eve_capacity"])))
                    inr.append(float(np.mean(info["bob_jammer_inr"])))
                    repairs.append(int(info["ris_action_repaired"]))
                    if terminated or truncated:
                        break
                total_steps += len(rewards)
                row = {
                    "episode": episode, "env_steps": total_steps, "policy_version": 0,
                    "return": float(np.sum(rewards)), "mean_reward": float(np.mean(rewards)),
                    "length": len(rewards), "outage_rate": float(info["episode_outage_rate"]),
                    "num_eves": 1 + int(info["num_passive_eves"]),
                    "mean_raw_security_gap": float(np.mean(bob) - np.mean(eve)),
                    "mean_bob_capacity": float(np.mean(bob)),
                    "mean_max_eve_capacity": float(np.mean(eve)),
                    "max_bob_capacity": float(np.mean(bob)),
                    "max_eve_capacity": float(np.mean(eve)),
                    "mean_bob_jammer_inr": float(np.mean(inr)),
                    "ris_repair_rate": float(np.mean(repairs)),
                }
                ew.writerow(row)
                for stream in (ef, sf, tf):
                    stream.flush()
                batch_episodes.append(row)
                all_episodes.append(row)
                plotter.on_episode(row)
                print(f"episode={episode:6d} return={row['return']:.4f} SOP={row['outage_rate']:.4f} bob_C={row['max_bob_capacity']:.6f} eve_C={row['max_eve_capacity']:.6f}", flush=True)
                if len(batch_episodes) == episodes_per_update or episode == episodes:
                    update += 1
                    metrics = dict.fromkeys(PPO_UPDATE_FIELDS)
                    metrics.update({
                        "update": update, "total_episodes": episode,
                        "total_env_steps": total_steps,
                        "mean_episode_return": float(np.mean([r["return"] for r in batch_episodes])),
                        "mean_outage_rate": float(np.mean([r["outage_rate"] for r in batch_episodes])),
                        **{"mean_" + key: float(np.mean([r[key] for r in batch_steps])) for key in ("initial_objective", "iterations", "evaluations", "solve_seconds")},
                        "mean_final_objective": float(np.mean([r["objective"] for r in batch_steps])),
                    })
                    uw.writerow(metrics)
                    uf.flush()
                    plotter.on_update(metrics)
                    write_json(run_dir / "checkpoint.json", {
                        **metadata, "completed_episodes": episode,
                        "total_env_steps": total_steps, "completed_batches": update,
                        "note": "Progress record, not a resumable RL checkpoint",
                    })
                    batch_episodes.clear()
                    batch_steps.clear()
            summary = {
                "algorithm": "AO", "information": "oracle_full_csi",
                "completed_episodes": episodes, "total_env_steps": total_steps,
                "completed_batches": update,
                **{key: float(np.mean([r[key] for r in all_episodes])) for key in ("outage_rate", "mean_bob_capacity", "mean_max_eve_capacity", "mean_raw_security_gap")},
            }
            write_json(run_dir / "summary.json", summary)
            return summary
    except BaseException:
        (run_dir / "training_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        plotter.finish()
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Full-CSI traditional AO reference; no neural-network training")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--episodes", type=int)
    budget.add_argument("--total-updates", "--total_updates", type=int, default=1000)
    parser.add_argument("--episodes-per-update", "--episodes_per_update", type=int, default=8)
    parser.add_argument("--ao-max-iterations", type=int, default=3)
    parser.add_argument("--ao-gradient-step", type=float, default=0.25)
    parser.add_argument("--ao-finite-difference", type=float, default=1e-3)
    args, system_argv = parser.parse_known_args(argv)
    episodes = args.episodes if args.episodes is not None else args.total_updates * args.episodes_per_update
    if episodes <= 0 or args.total_updates <= 0 or args.episodes_per_update <= 0:
        parser.error("episode counts and total updates must be positive")
    config = AOConfig(max_iterations=args.ao_max_iterations, gradient_step=args.ao_gradient_step, finite_difference=args.ao_finite_difference)
    params = get_system_params(system_argv)
    run_dir = Path(__file__).resolve().parent / "result" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    return run_experiment(params, config, episodes, args.episodes_per_update, run_dir)


if __name__ == "__main__":
    main()
