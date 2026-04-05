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
    KITCHEN_IMAGE_ALL, KITCHEN_IMAGE_COMBINED_DATAPATHS,
    KITCHEN_IMAGE_COMPLETE_DATAPATH,
)
from policy_extraction.methods import AWRTrainer, GradualAWRTrainer
from policy_extraction.evaluate import (
    SUBTASK_BOUNDARIES, get_subtask_configs, rollout_image_d4rl, evaluate_policy,
)


@hydra.main(config_path="config", config_name="policy_extraction_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.device)
    output_dir = Path(os.getcwd())

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Build trainer first — determines image vs state mode from checkpoint
    if cfg.method == "awr":
        trainer = AWRTrainer(cfg.snapshot, cfg.temperature, cfg.max_weight, device)
    elif cfg.method == "gradual_awr":
        trainer = GradualAWRTrainer(cfg.snapshot, cfg.temperature, cfg.max_weight, cfg.warmup_exponent, device)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    # Load matching dataset
    if trainer.is_image:
        assert cfg.datapath, "Image model requires datapath to be set"
        dataset = ImagePolicyExtractionBuffer(
            datapath=cfg.datapath,
            num_goal_samples=cfg.num_goal_samples,
            seed=cfg.seed,
            frame_stack=cfg.frame_stack,
        )
    else:
        assert cfg.dataset, "State model requires dataset to be set"
        dataset = PolicyExtractionBuffer(
            datasource=cfg.dataset,
            num_goal_samples=cfg.num_goal_samples,
            seed=cfg.seed,
        )

    # Build policy
    trainable_encoder = cfg.get("trainable_encoder", False) and trainer.is_image
    if trainable_encoder:
        policy = CNNGaussianPolicy(
            action_dim=9,
            frame_stack=cfg.frame_stack,
            encoder_size=cfg.get("encoder_size", 18),
            encoder_dim=cfg.get("encoder_dim", 256),
            hidden_dims=list(cfg.policy_hidden_dims),
            log_std_init=cfg.log_std_init,
        ).to(device)
    else:
        input_dim = 2 * trainer.hidden_dim if trainer.is_image else 2 * 59
        policy = GaussianMLPPolicy(
            input_dim=input_dim,
            action_dim=9,
            hidden_dims=list(cfg.policy_hidden_dims),
            log_std_init=cfg.log_std_init,
        ).to(device)

    losses = trainer.train(dataset, policy, cfg.epochs, cfg.batch_size, cfg.lr,
                           trainable_encoder=trainable_encoder)
    policy.eval()

    # Save checkpoint
    policy_path = output_dir / "policy.pt"
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "snapshot": cfg.snapshot,
        "config": OmegaConf.to_container(cfg),
    }, policy_path)
    print(f"Saved policy to {policy_path}")

    # Plot losses
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "training_losses.png", dpi=150)
    plt.close()

    results = {"final_loss": losses[-1], "subtask_results": []}

    if not cfg.eval.enabled:
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    subtask_results = []

    if trainer.is_image:
        import gym
        import d4rl
        traj0_dir = os.path.join(KITCHEN_IMAGE_COMPLETE_DATAPATH, "traj0")
        d4rl_env = gym.make('kitchen-complete-v0')
        all_obs = d4rl_env.get_dataset()['observations']  # (N, 60), [:30] = qpos
        for st in SUBTASK_BOUNDARIES:
            init_t, goal_t = st['init_t'], st['goal_t']
            goal_frames = [
                Image.open(os.path.join(traj0_dir, f"img{max(0, goal_t - cfg.frame_stack + 1 + k)}.png"))
                for k in range(cfg.frame_stack)
            ]
            init_qpos_seq = [all_obs[max(0, init_t - cfg.frame_stack + 1 + k), :30]
                            for k in range(cfg.frame_stack)]
            eval_model = None if trainable_encoder else trainer.model
            st_rewards = []
            for _ in range(cfg.eval.n_episodes):
                r = rollout_image_d4rl(
                    d4rl_env, policy, eval_model,
                    init_qpos_seq, goal_frames, st['max_steps'], device,
                )
                st_rewards.append(r["total_reward"])
            baseline = st['baseline_tasks']
            successes = [int(tr > baseline) for tr in st_rewards]
            sr = float(np.mean(successes))
            subtask_results.append({
                "subtask_idx":    st['subtask_idx'],
                "subtask_name":   st['subtask_name'],
                "success_rate":   sr,
                "mean_tasks_completed": float(np.mean(st_rewards)),
                "baseline_tasks": baseline,
            })
            print(f"  [{st['subtask_name']:>14}]  success={sr:.2f}  "
                  f"mean_tasks={np.mean(st_rewards):.2f}  (baseline: {baseline})")
        d4rl_env.close()

    else:
        import minari
        eval_dataset = "D4RL/kitchen/complete-v2"
        subtask_boundaries = get_subtask_configs(eval_dataset)
        dataset = minari.load_dataset(eval_dataset, download=True)
        env = dataset.recover_environment(robot_noise_ratio=0.0, object_noise_ratio=0.0)
        for st in subtask_boundaries:
            st_results = evaluate_policy(
                env, policy, st['goal_obs'], st['init_obs'],
                n_episodes=cfg.eval.n_episodes,
                max_steps=st['max_steps'],
                device=device,
            )
            baseline = st['baseline_tasks']
            successes = [int(tc > baseline) for tc in st_results["per_episode_tasks"]]
            sr = float(np.mean(successes))
            subtask_results.append({
                "subtask_idx":    st['subtask_idx'],
                "subtask_name":   st['subtask_name'],
                "success_rate":   sr,
                "mean_tasks_completed": st_results["mean_tasks_completed"],
                "baseline_tasks": baseline,
            })
            print(f"  [{st['subtask_name']:>14}]  success={sr:.2f}  "
                  f"mean_tasks={st_results['mean_tasks_completed']:.2f}  (baseline: {baseline})")
        env.close()

    mean_sr = float(np.mean([s["success_rate"] for s in subtask_results]))
    print(f"\n  Mean subtask success rate: {mean_sr:.2f}")
    results["subtask_results"] = subtask_results
    results["mean_success_rate"] = mean_sr

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
