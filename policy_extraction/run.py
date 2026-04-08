"""
Policy extraction from frozen VIP value functions using AWR.

Usage:
    # State model
    python -m policy_extraction.run \
        snapshot=/path/to/snapshot_500000.pt \
        dataset=D4RL/kitchen/all-v2

    # Image model (datapath basename must be a valid d4rl env name, e.g. kitchen-complete-v0)
    python -m policy_extraction.run \
        snapshot=/path/to/snapshot_50000.pt \
        datapath=/path/to/kitchen-complete-v0 \
        frame_stack=1

    # Multi-seed
    python -m policy_extraction.run \
        snapshot=/path/to/snapshot.pt \
        dataset=D4RL/kitchen/all-v2 \
        seeds='[42,43,44]'
"""
import json
import os
import numpy as np
import torch
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from PIL import Image

from policy_extraction.policy import GaussianMLPPolicy, CNNGaussianPolicy
from policy_extraction.data import (
    PolicyExtractionBuffer, ImagePolicyExtractionBuffer,
)
from policy_extraction.methods import AWRTrainer, GradualAWRTrainer, BCTrainer, RepTransferTrainer
from policy_extraction.evaluate import (
    get_subtask_configs, get_full_sequence_config,
    get_image_subtask_configs, get_image_full_sequence_configs,
    rollout_image_d4rl, evaluate_policy, KITCHEN_TASKS,
)


def _build_policy(cfg, trainer, device):
    """Create a fresh, untrained policy."""
    if trainer.use_pretrained:
        return CNNGaussianPolicy(
            action_dim=9,
            frame_stack=cfg.frame_stack,
            encoder_size=cfg.get("encoder_size", 18),
            hidden_dims=list(cfg.policy_hidden_dims),
            log_std_init=cfg.log_std_init,
        ).to(device)
    else:
        input_dim = 2 * trainer.hidden_dim if trainer.uses_embeddings else 2 * 59
        return GaussianMLPPolicy(
            input_dim=input_dim,
            action_dim=9,
            hidden_dims=list(cfg.policy_hidden_dims),
            log_std_init=cfg.log_std_init,
        ).to(device)


def _save_policy(policy, path, cfg, seed, policy_name):
    """Save a policy checkpoint."""
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "snapshot": cfg.snapshot,
        "seed": seed,
        "policy_name": policy_name,
        "config": OmegaConf.to_container(cfg),
    }, path)
    print(f"  Saved {policy_name} policy to {path}")


def _create_dataset(cfg, trainer, seed, noise_ratio, subtask_name=None):
    """Create the appropriate buffer for state or image models."""
    noise_dir = cfg.get("noise_dir", "")
    if trainer.is_image:
        assert cfg.datapath, "Image model requires datapath to be set"
        return ImagePolicyExtractionBuffer(
            datapath=cfg.datapath,
            num_goal_transitions=cfg.num_goal_samples,
            seed=seed,
            frame_stack=cfg.frame_stack,
            subtask_name=subtask_name,
            noise_ratio=noise_ratio,
            noise_dir=noise_dir,
        )
    else:
        assert cfg.dataset, "State model requires dataset to be set"
        return PolicyExtractionBuffer(
            datasource=cfg.dataset,
            num_goal_transitions=cfg.num_goal_samples,
            seed=seed,
            subtask_name=subtask_name,
            noise_ratio=noise_ratio,
            noise_dir=noise_dir,
        )


def run_seed(cfg, seed, noise_ratio, trainer, device, output_dir):
    """Train and evaluate per-subtask + full-sequence policies for a single seed."""
    print(f"\n{'='*60}")
    print(f"  Seed {seed}  noise_ratio={noise_ratio}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    policies = {}
    all_losses = {}

    # --- Train subtask policies ---
    for task_name in KITCHEN_TASKS:
        print(f"\n--- Training policy: {task_name} ---")
        dataset = _create_dataset(cfg, trainer, seed, noise_ratio, subtask_name=task_name)
        policy = _build_policy(cfg, trainer, device)
        losses = trainer.train(dataset, policy, cfg.epochs, cfg.batch_size, cfg.lr)
        policy.eval()
        policies[task_name] = policy
        all_losses[task_name] = losses
        if cfg.save_policies:
            _save_policy(policy, output_dir / f"policy_{task_name.replace(' ', '_')}_seed{seed}.pt",
                         cfg, seed, task_name)

    # --- Train full-sequence policy (NO filtering, all episodes, all transitions) ---
    print(f"\n--- Training policy: full_sequence ---")
    dataset = _create_dataset(cfg, trainer, seed, noise_ratio, subtask_name=None)
    policy = _build_policy(cfg, trainer, device)
    losses = trainer.train(dataset, policy, cfg.epochs, cfg.batch_size, cfg.lr)
    policy.eval()
    policies["full_sequence"] = policy
    all_losses["full_sequence"] = losses
    if cfg.save_policies:
        _save_policy(policy, output_dir / f"policy_full_sequence_seed{seed}.pt",
                     cfg, seed, "full_sequence")

    results = {"seed": seed, "policies": {}}
    for name, loss_list in all_losses.items():
        results["policies"][name] = {"final_loss": loss_list[-1]}

    if not cfg.eval.enabled:
        return results, policies, all_losses

    # --- Subtask eval (each subtask uses its own policy) ---
    print("\n  Subtask evaluation...")
    subtask_results = []

    if trainer.is_image:
        import gym
        import d4rl
        d4rl_env_name = os.path.basename(cfg.datapath.rstrip("/"))
        d4rl_env = gym.make(d4rl_env_name)
        image_subtask_configs = get_image_subtask_configs(
            cfg.datapath, os.path.basename(cfg.datapath.rstrip("/")),
            frame_stack=cfg.frame_stack,
        )
        eval_model = None if trainer.use_pretrained else trainer.model
        for task_name in KITCHEN_TASKS:
            eval_states = image_subtask_configs[task_name]
            st_rewards = []
            for es in eval_states:
                r = rollout_image_d4rl(
                    d4rl_env, policies[task_name], eval_model,
                    es['init_qpos_seq'], es['goal_frames'],
                    es['max_steps'], device,
                )
                st_rewards.append(r["total_reward"])
            baseline = eval_states[0]['baseline_tasks']
            successes = [int(tr > baseline) for tr in st_rewards]
            sr = float(np.mean(successes))
            subtask_results.append({
                "subtask_name":   task_name,
                "success_rate":   sr,
                "mean_tasks_completed": float(np.mean(st_rewards)),
                "std_tasks_completed": float(np.std(st_rewards)),
                "baseline_tasks": baseline,
                "n_eval_states":  len(eval_states),
            })
            print(f"  [{task_name:>14}]  success={sr:.2f}  "
                  f"mean_tasks={np.mean(st_rewards):.2f}  "
                  f"(baseline: {baseline}, n={len(eval_states)})")

    else:
        import minari
        eval_dataset = "D4RL/kitchen/complete-v2"
        subtask_configs = get_subtask_configs(eval_dataset)
        eval_ds = minari.load_dataset(eval_dataset, download=True)
        env = eval_ds.recover_environment(robot_noise_ratio=0.0, object_noise_ratio=0.0)
        for task_name in KITCHEN_TASKS:
            eval_states = subtask_configs[task_name]
            st_eval = evaluate_policy(
                env, policies[task_name], eval_states, device=device,
            )
            baseline = eval_states[0]['baseline_tasks']
            successes = [int(tc > baseline) for tc in st_eval["per_episode_tasks"]]
            sr = float(np.mean(successes))
            subtask_results.append({
                "subtask_name":   task_name,
                "success_rate":   sr,
                "mean_tasks_completed": st_eval["mean_tasks_completed"],
                "std_tasks_completed": st_eval["std_tasks_completed"],
                "baseline_tasks": baseline,
                "n_eval_states":  len(eval_states),
            })
            print(f"  [{task_name:>14}]  success={sr:.2f}  "
                  f"mean_tasks={st_eval['mean_tasks_completed']:.2f}  "
                  f"(baseline: {baseline}, n={len(eval_states)})")

    if subtask_results:
        mean_sr = float(np.mean([s["success_rate"] for s in subtask_results]))
        results["subtask_results"] = {
            "per_subtask": subtask_results,
            "mean_success_rate": mean_sr,
        }
        print(f"\n  Mean subtask success rate: {mean_sr:.2f}")

    # --- Full-sequence eval ---
    print("\n  Full-sequence evaluation...")
    if trainer.is_image:
        fs_image_configs = get_image_full_sequence_configs(
            cfg.datapath, os.path.basename(cfg.datapath.rstrip("/")),
            frame_stack=cfg.frame_stack,
        )
        fs_rewards = []
        for es in fs_image_configs:
            r = rollout_image_d4rl(
                d4rl_env, policies["full_sequence"], eval_model,
                es['init_qpos_seq'], es['goal_frames'],
                es['max_steps'], device,
            )
            fs_rewards.append(r["total_reward"])
        results["policies"]["full_sequence"]["full_seq_eval"] = {
            "mean_tasks_completed": float(np.mean(fs_rewards)),
            "std_tasks_completed": float(np.std(fs_rewards)),
            "per_episode_tasks": [float(r) for r in fs_rewards],
        }
        d4rl_env.close()
    else:
        fs_eval_states = get_full_sequence_config()
        fs_results = evaluate_policy(
            env, policies["full_sequence"], fs_eval_states, device=device,
        )
        results["policies"]["full_sequence"]["full_seq_eval"] = {
            "mean_tasks_completed": fs_results["mean_tasks_completed"],
            "std_tasks_completed": fs_results["std_tasks_completed"],
            "per_episode_tasks": fs_results["per_episode_tasks"],
        }
        env.close()
    fs_mtc = results["policies"]["full_sequence"]["full_seq_eval"]["mean_tasks_completed"]
    print(f"  [full sequence]  mean_tasks={fs_mtc:.2f}/4")

    return results, policies, all_losses


def _make_trainer(method, mcfg, device):
    """Instantiate a trainer for the given method name."""
    if method == "awr":
        return AWRTrainer(mcfg.snapshot, mcfg.temperature, mcfg.max_weight, device)
    elif method == "gradual_awr":
        return GradualAWRTrainer(mcfg.snapshot, mcfg.temperature, mcfg.max_weight, mcfg.warmup_exponent, device)
    elif method == "bc":
        return BCTrainer(mcfg.snapshot, device)
    elif method == "rep_transfer":
        return RepTransferTrainer(mcfg.snapshot, device)
    else:
        raise ValueError(f"Unknown method: {method}")


def _run_model(cfg, model_cfg, device, model_dir):
    """Run all methods, noise ratios and seeds for a single model."""
    # Overlay model-specific fields onto base config
    OmegaConf.set_struct(cfg, False)
    mcfg = OmegaConf.merge(cfg, {
        "snapshot": model_cfg.snapshot,
        "dataset": model_cfg.get("dataset", ""),
        "datapath": model_cfg.get("datapath", ""),
        "frame_stack": model_cfg.get("frame_stack", 1),
        "batch_size": model_cfg.get("batch_size", 256),
    })

    for method in mcfg.methods:
        print(f"\n{'+'*60}")
        print(f"  Method: {method}")
        print(f"{'+'*60}")
        trainer = _make_trainer(method, mcfg, device)
        method_dir = model_dir / method
        method_dir.mkdir(exist_ok=True)

        for noise_ratio in mcfg.noise_ratios:
            nr_dir = method_dir / f"noise_{noise_ratio}"
            nr_dir.mkdir(exist_ok=True)
            print(f"\n{'#'*60}")
            print(f"  noise_ratio={noise_ratio}")
            print(f"{'#'*60}")

            all_seed_results = []
            all_seed_losses = []  # list of dicts: {policy_name: [loss_per_epoch]}
            for seed in mcfg.seeds:
                seed_results, policies, seed_losses = run_seed(
                    mcfg, seed, noise_ratio, trainer, device, nr_dir)
                all_seed_results.append(seed_results)
                all_seed_losses.append(seed_losses)

            # Plot losses: one subplot per policy, one line per seed
            policy_names = list(all_seed_losses[0].keys())
            n_policies = len(policy_names)
            fig, axes = plt.subplots(1, n_policies, figsize=(5 * n_policies, 4), squeeze=False)
            for pi, pname in enumerate(policy_names):
                ax = axes[0, pi]
                for i, (seed, seed_losses) in enumerate(zip(mcfg.seeds, all_seed_losses)):
                    if pname in seed_losses:
                        ax.plot(seed_losses[pname], label=f"seed {seed}", alpha=0.7)
                ax.set_title(pname)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                if len(mcfg.seeds) > 1:
                    ax.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(nr_dir / "training_losses.png", dpi=150)
            plt.close()

            # Aggregate results
            if len(mcfg.seeds) == 1:
                results = all_seed_results[0]
            else:
                results = {"per_seed": all_seed_results}

                # Aggregate subtask results across seeds
                first_st = all_seed_results[0].get("subtask_results")
                if mcfg.eval.enabled and first_st:
                    per_subtask_list = first_st["per_subtask"]
                    n_subtasks = len(per_subtask_list)
                    agg_subtasks = []
                    for si in range(n_subtasks):
                        srs = [r["subtask_results"]["per_subtask"][si]["success_rate"]
                               for r in all_seed_results
                               if si < len(r.get("subtask_results", {}).get("per_subtask", []))]
                        mtcs = [r["subtask_results"]["per_subtask"][si]["mean_tasks_completed"]
                                for r in all_seed_results
                                if si < len(r.get("subtask_results", {}).get("per_subtask", []))]
                        ref = per_subtask_list[si]
                        agg_subtasks.append({
                            "subtask_name":   ref["subtask_name"],
                            "mean_success_rate": float(np.mean(srs)),
                            "std_success_rate":  float(np.std(srs)),
                            "mean_tasks_completed": float(np.mean(mtcs)),
                            "std_tasks_completed":  float(np.std(mtcs)),
                            "baseline_tasks": ref["baseline_tasks"],
                        })

                    all_mean_srs = [r["subtask_results"]["mean_success_rate"] for r in all_seed_results
                                    if "subtask_results" in r]
                    results["subtask_results"] = {
                        "per_subtask": agg_subtasks,
                        "mean_success_rate": float(np.mean(all_mean_srs)),
                        "std_success_rate": float(np.std(all_mean_srs)),
                    }

                # Aggregate full-sequence results across seeds
                fs_mtcs = [r["policies"]["full_sequence"]["full_seq_eval"]["mean_tasks_completed"]
                           for r in all_seed_results
                           if "full_seq_eval" in r.get("policies", {}).get("full_sequence", {})]
                if fs_mtcs:
                    results["full_sequence"] = {
                        "mean_tasks_completed": float(np.mean(fs_mtcs)),
                        "std_tasks_completed":  float(np.std(fs_mtcs)),
                    }

                # Print aggregate summary
                print(f"\n{'='*60}")
                print(f"  AGGREGATE RESULTS (noise={noise_ratio}, {len(mcfg.seeds)} seeds)")
                print(f"{'='*60}")
                if "subtask_results" in results:
                    for st in results["subtask_results"]["per_subtask"]:
                        print(f"  [{st['subtask_name']:>14}]  success={st['mean_success_rate']:.2f}±{st['std_success_rate']:.2f}  "
                              f"mean_tasks={st['mean_tasks_completed']:.2f}±{st['std_tasks_completed']:.2f}")
                    sr = results["subtask_results"]
                    print(f"\n  Mean success rate: {sr['mean_success_rate']:.2f}±{sr['std_success_rate']:.2f}")
                if "full_sequence" in results:
                    print(f"  [full sequence]  mean_tasks={results['full_sequence']['mean_tasks_completed']:.2f}"
                          f"±{results['full_sequence']['std_tasks_completed']:.2f}/4")

            results["noise_ratio"] = noise_ratio
            with open(nr_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {nr_dir}")


@hydra.main(config_path="config", config_name="policy_extraction_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.device)
    output_dir = Path(os.getcwd())

    for model_cfg in cfg.models:
        model_name = model_cfg.name
        model_dir = output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        print(f"\n{'*'*60}")
        print(f"  Model: {model_name}  ({model_cfg.snapshot})")
        print(f"{'*'*60}")
        _run_model(cfg, model_cfg, device, model_dir)


if __name__ == "__main__":
    main()
