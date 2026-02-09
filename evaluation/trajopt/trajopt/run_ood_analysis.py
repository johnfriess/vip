"""
Diagnostic script to analyze why the VIP value function is out-of-distribution.

Two analyses:
1. Sample N trajectories from MPPI and compute statistics on scores/states
   to determine if the value function is noisy vs systematically biased.
2. Compare sampled trajectory states against the training dataset to
   identify which states are OOD and whether closer-to-goal states exist
   in the data.

Usage:
    python run_ood_analysis.py \
        checkpoint_path=/path/to/snapshot_200000.pt \
        [n_trajectories=100] [plan_horizon=12]
"""
import os
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
import minari
import multiprocessing as mp
from matplotlib import pyplot as plt

from PIL import Image
from trajopt.envs.obs_wrappers import env_constructor
from trajopt.algos.mppi import MPPI
from trajopt.utils.utils import generate_perturbed_actions, do_env_rollout


def load_training_data(datasource="D4RL/kitchen/complete-v2"):
    """Load all trajectories from the training dataset."""
    dataset = minari.load_dataset(datasource, download=True)
    all_obs = []
    episode_obs = []
    for ep in dataset.iterate_episodes():
        obs = ep.observations['observation']
        if len(obs) > 2:
            all_obs.append(obs.astype(np.float32))
            episode_obs.append(obs.astype(np.float32))
    all_obs_flat = np.concatenate(all_obs, axis=0)  # (N_total, 59)
    return all_obs_flat, episode_obs


def compute_embedding_stats(model, states, goal_state, device='cpu'):
    """Compute embeddings and distances to goal for a batch of states."""
    with torch.no_grad():
        states_t = torch.tensor(states, dtype=torch.float32).to(device)
        goal_t = torch.tensor(goal_state, dtype=torch.float32).unsqueeze(0).to(device)
        emb_states = model(states_t).cpu().numpy()
        emb_goal = model(goal_t).cpu().numpy()
    distances = np.linalg.norm(emb_states - emb_goal, axis=-1)
    return emb_states, distances


@hydra.main(config_name="mppi_config", config_path="config")
def main(cfg: DictConfig):
    OUT_DIR = '.'
    N_TRAJ = cfg.get('n_trajectories', 100)  # override: n_trajectories=100

    # --- Setup environment and model (same as run_mppi.py) ---
    if "state_" in cfg.embedding:
        cfg.env_kwargs.load_path = f"{cfg.embedding}:{cfg.checkpoint_path}"
    else:
        cfg.env_kwargs.load_path = cfg.embedding

    env_kwargs = cfg['env_kwargs']
    env = env_constructor(**env_kwargs)

    mean = np.zeros(env.action_dim)
    sigma = 1.0 * np.ones(env.action_dim)
    filter_coefs = [sigma, cfg['filter']['beta_0'], cfg['filter']['beta_1'], cfg['filter']['beta_2']]

    seed = cfg['seed']
    env.reset(seed=seed)

    agent = MPPI(env,
                 H=cfg['plan_horizon'],
                 paths_per_cpu=N_TRAJ,  # Sample N trajectories
                 num_cpu=1,
                 kappa=cfg['kappa'],
                 gamma=cfg['gamma'],
                 mean=mean,
                 filter_coefs=filter_coefs,
                 default_act=cfg['default_act'],
                 seed=seed,
                 env_kwargs=env_kwargs)

    # Get the model and goal embedding for analysis
    # Wrapper chain: GymEnv.env -> StateEmbedding (has .embedding, .goal_embedding, .device)
    state_emb_wrapper = env.env
    embedding_model = state_emb_wrapper.embedding  # StateVIP model (DataParallel)
    goal_embedding = state_emb_wrapper.goal_embedding[cfg.camera]
    device = state_emb_wrapper.device

    # Get goal state from minari
    minari_dataset = minari.load_dataset(cfg.env, download=True)
    episodes = list(minari_dataset.iterate_episodes())
    goal_ep = episodes[cfg.goal_episode]
    goal_obs = goal_ep.observations['observation']
    goal_state = goal_obs[min(cfg.goal_timestep, len(goal_obs) - 1)].astype(np.float32)

    # --- Render init and goal states ---
    os.makedirs("ood_analysis", exist_ok=True)
    saved_state = env.get_env_state()

    # Build state dicts from 59D observations (same layout as obs_wrappers.py init_state)
    init_obs = goal_obs[min(cfg.init_timestep, len(goal_obs) - 1)].astype(np.float32)
    init_state_dict = {
        "qpos": np.concatenate([init_obs[:9], init_obs[18:39]]),
        "qvel": np.concatenate([init_obs[9:18], init_obs[39:59]]),
    }
    goal_state_dict = {
        "qpos": np.concatenate([goal_state[:9], goal_state[18:39]]),
        "qvel": np.concatenate([goal_state[9:18], goal_state[39:59]]),
    }

    camera = cfg.camera

    # Render init and goal, save side-by-side comparison
    state_emb_wrapper.set_env_state(init_state_dict)
    init_img = state_emb_wrapper.env.get_image(camera_name=camera)

    state_emb_wrapper.set_env_state(goal_state_dict)
    goal_img = state_emb_wrapper.env.get_image(camera_name=camera)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(init_img)
    axes[0].set_title(f"Init (timestep {cfg.init_timestep})")
    axes[0].axis('off')
    axes[1].imshow(goal_img)
    axes[1].set_title(f"Goal (timestep {cfg.goal_timestep})")
    axes[1].axis('off')
    plt.suptitle(f"Episode {cfg.goal_episode} | {cfg.env}", fontsize=13)
    plt.tight_layout()
    plt.savefig("ood_analysis/init_vs_goal.png", dpi=150)
    plt.close()
    print("Saved: ood_analysis/init_vs_goal.png")

    # Restore env state
    env.set_env_state(saved_state)

    print("=" * 70)
    print("ANALYSIS 1: Trajectory Sampling Statistics")
    print("=" * 70)

    # --- Analysis 1: Sample N trajectories and compute statistics ---
    # Do one round of MPPI rollouts with N_TRAJ trajectories
    paths = agent.do_rollouts(seed)
    scores = agent.score_trajectory(paths)

    print(f"\nSampled {len(paths)} trajectories (horizon={cfg.plan_horizon})")
    print(f"  Score (terminal reward = -||φ(s_T) - φ(g)||):")
    print(f"    mean  = {scores.mean():.4f}")
    print(f"    std   = {scores.std():.4f}")
    print(f"    min   = {scores.min():.4f}")
    print(f"    max   = {scores.max():.4f}")
    print(f"    median= {np.median(scores):.4f}")

    # Coefficient of variation: high = noisy, low = systematic
    cv = scores.std() / (abs(scores.mean()) + 1e-8)
    print(f"    CV (std/|mean|) = {cv:.4f}")
    if cv > 1.0:
        print("    >> HIGH variance relative to mean -> value function is NOISY")
    elif cv < 0.3:
        print("    >> LOW variance -> trajectories are similar, systematic bias likely")
    else:
        print("    >> MODERATE variance")

    # Percentile analysis
    pcts = [10, 25, 50, 75, 90]
    print(f"  Percentiles: {dict(zip(pcts, [f'{np.percentile(scores, p):.4f}' for p in pcts]))}")

    # Check how many trajectories have positive vs negative trend
    improving_count = 0
    for path in paths:
        rews = path['rewards']
        if len(rews) >= 2 and rews[-1] > rews[0]:
            improving_count += 1
    print(f"  Trajectories improving over horizon: {improving_count}/{len(paths)} "
          f"({100*improving_count/len(paths):.1f}%)")

    # --- Collect terminal states from all sampled trajectories ---
    terminal_obs = []
    all_step_obs = []
    for path in paths:
        terminal_obs.append(path['observations'][-1])
        for obs in path['observations']:
            all_step_obs.append(obs)
    terminal_obs = np.array(terminal_obs)  # (N, 59)
    all_step_obs = np.array(all_step_obs)  # (N*H, 59)

    # Compute raw state-space distance to goal (not embedding)
    raw_distances_terminal = np.linalg.norm(terminal_obs - goal_state, axis=-1)
    print(f"\n  Raw state-space distance (terminal states to goal):")
    print(f"    mean={raw_distances_terminal.mean():.4f}  std={raw_distances_terminal.std():.4f}")
    print(f"    min={raw_distances_terminal.min():.4f}  max={raw_distances_terminal.max():.4f}")

    # Compute embedding distances for terminal states
    emb_terminal, emb_dist_terminal = compute_embedding_stats(
        embedding_model.module, terminal_obs, goal_state, device)
    print(f"\n  Embedding distance (terminal states to goal):")
    print(f"    mean={emb_dist_terminal.mean():.4f}  std={emb_dist_terminal.std():.4f}")
    print(f"    min={emb_dist_terminal.min():.4f}  max={emb_dist_terminal.max():.4f}")

    # Check correlation between raw distance and embedding distance
    correlation = np.corrcoef(raw_distances_terminal, emb_dist_terminal)[0, 1]
    print(f"\n  Correlation(raw_dist, emb_dist) = {correlation:.4f}")
    if correlation > 0.7:
        print("    >> Good: embedding preserves distance structure")
    elif correlation > 0.3:
        print("    >> Moderate: embedding partially preserves distances")
    else:
        print("    >> POOR: embedding does NOT preserve distance structure!")
        print("       This is a key sign the value function is unreliable OOD")

    print("\n" + "=" * 70)
    print("ANALYSIS 2: Comparison with Training Data Distribution")
    print("=" * 70)

    # --- Analysis 2: Compare against training data ---
    print(f"\nLoading training data from {cfg.env}...")
    train_obs_flat, train_episodes = load_training_data(cfg.env)
    print(f"  Training data: {len(train_episodes)} episodes, {len(train_obs_flat)} total states")

    # Compute training data statistics
    train_mean = train_obs_flat.mean(axis=0)
    train_std = train_obs_flat.std(axis=0) + 1e-8
    train_min = train_obs_flat.min(axis=0)
    train_max = train_obs_flat.max(axis=0)

    # Check how many sampled states fall outside training bounds
    init_obs = agent.sol_obs[0]  # Initial state
    print(f"\n  Initial state raw distance to goal: {np.linalg.norm(init_obs - goal_state):.4f}")

    # For terminal states: compute per-dimension deviation from training range
    beyond_min = (terminal_obs < train_min[None, :]).sum(axis=1)
    beyond_max = (terminal_obs > train_max[None, :]).sum(axis=1)
    n_ood_dims = beyond_min + beyond_max
    print(f"\n  Terminal states outside training data bounds:")
    print(f"    Avg dims out of range: {n_ood_dims.mean():.1f} / {terminal_obs.shape[1]}")
    print(f"    Max dims out of range: {n_ood_dims.max()} / {terminal_obs.shape[1]}")
    print(f"    States with ANY dim OOD: {(n_ood_dims > 0).sum()}/{len(terminal_obs)}")

    # Mahalanobis-like distance: how many std devs away from training mean
    z_scores_terminal = np.abs((terminal_obs - train_mean) / train_std)
    print(f"\n  Z-score (std devs from training mean) per state:")
    print(f"    Mean max z-score across dims: {z_scores_terminal.max(axis=1).mean():.2f}")
    print(f"    Worst z-score: {z_scores_terminal.max():.2f}")

    # For ALL sampled states (not just terminal):
    z_scores_all = np.abs((all_step_obs - train_mean) / train_std)
    print(f"\n  Z-score for ALL sampled states (across full horizon):")
    print(f"    Mean max z-score: {z_scores_all.max(axis=1).mean():.2f}")
    print(f"    Fraction with any dim > 3 std: "
          f"{(z_scores_all.max(axis=1) > 3).mean()*100:.1f}%")
    print(f"    Fraction with any dim > 5 std: "
          f"{(z_scores_all.max(axis=1) > 5).mean()*100:.1f}%")

    # Nearest-neighbor distance: for each terminal state, find closest training state
    print(f"\n  Nearest-neighbor analysis (may be slow for large datasets)...")
    # Subsample training data if too large
    if len(train_obs_flat) > 10000:
        subsample_idx = np.random.choice(len(train_obs_flat), 10000, replace=False)
        train_subsample = train_obs_flat[subsample_idx]
    else:
        train_subsample = train_obs_flat

    nn_dists = []
    for t_obs in terminal_obs:
        dists = np.linalg.norm(train_subsample - t_obs, axis=-1)
        nn_dists.append(dists.min())
    nn_dists = np.array(nn_dists)
    print(f"    NN distance (terminal -> nearest training state):")
    print(f"      mean={nn_dists.mean():.4f}  std={nn_dists.std():.4f}")
    print(f"      min={nn_dists.min():.4f}  max={nn_dists.max():.4f}")

    # Compare: what is the NN distance for states WITHIN the dataset?
    intra_nn_dists = []
    sample_idx = np.random.choice(len(train_subsample), min(100, len(train_subsample)), replace=False)
    for idx in sample_idx:
        dists = np.linalg.norm(train_subsample - train_subsample[idx], axis=-1)
        dists[idx] = np.inf  # exclude self
        intra_nn_dists.append(dists.min())
    intra_nn_dists = np.array(intra_nn_dists)
    print(f"    NN distance (within training data for reference):")
    print(f"      mean={intra_nn_dists.mean():.4f}  std={intra_nn_dists.std():.4f}")

    ood_ratio = nn_dists.mean() / (intra_nn_dists.mean() + 1e-8)
    print(f"    OOD ratio (sampled/training NN dist): {ood_ratio:.2f}x")
    if ood_ratio > 3:
        print("    >> Sampled states are FAR from training data -> strongly OOD")
    elif ood_ratio > 1.5:
        print("    >> Sampled states are moderately OOD")
    else:
        print("    >> Sampled states are near training distribution")

    # --- Check: do training episodes that get closer to goal exist? ---
    print(f"\n  Training trajectories approaching the goal:")
    train_emb_goal, _ = compute_embedding_stats(
        embedding_model.module, goal_state[None, :], goal_state, device)

    closest_train_dist = np.inf
    closest_ep_idx = -1
    for i, ep_obs in enumerate(train_episodes):
        _, ep_emb_dists = compute_embedding_stats(
            embedding_model.module, ep_obs, goal_state, device)
        if ep_emb_dists.min() < closest_train_dist:
            closest_train_dist = ep_emb_dists.min()
            closest_ep_idx = i

    print(f"    Closest training state to goal (embedding): {closest_train_dist:.4f} (episode {closest_ep_idx})")
    print(f"    Best sampled terminal state to goal (embedding): {emb_dist_terminal.min():.4f}")
    if emb_dist_terminal.min() < closest_train_dist:
        print("    >> MPPI found states CLOSER to goal than any training state!")
        print("       These are likely OOD states where the value function is extrapolating")
    else:
        print("    >> Sampled states are farther than best training state")

    # --- Generate plots ---
    print(f"\n{'='*70}")
    print("Generating diagnostic plots...")
    print(f"{'='*70}")
    # Plot 1: Score distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(scores, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(scores.mean(), color='r', linestyle='--', label=f'mean={scores.mean():.3f}')
    axes[0].set_xlabel('Terminal Value V(s_T, g)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Score Distribution ({N_TRAJ} trajectories)')
    axes[0].legend()

    # Plot 2: Raw distance vs embedding distance
    axes[1].scatter(raw_distances_terminal, emb_dist_terminal, alpha=0.5, s=10)
    axes[1].set_xlabel('Raw State Distance to Goal')
    axes[1].set_ylabel('Embedding Distance to Goal')
    axes[1].set_title(f'Raw vs Embedding Distance (corr={correlation:.3f})')

    # Plot 3: NN distance histogram
    axes[2].hist(nn_dists, bins=30, alpha=0.7, label='Sampled -> Train NN', edgecolor='black')
    axes[2].hist(intra_nn_dists, bins=30, alpha=0.5, label='Train -> Train NN', edgecolor='black')
    axes[2].set_xlabel('Nearest Neighbor Distance')
    axes[2].set_ylabel('Count')
    axes[2].set_title('OOD Detection: NN Distances')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("ood_analysis/diagnostic_plots.png", dpi=150)
    plt.close()
    print("  Saved: ood_analysis/diagnostic_plots.png")

    # Plot 4: Per-dimension OOD analysis
    fig, ax = plt.subplots(figsize=(14, 5))
    ood_frac_per_dim = ((terminal_obs < train_min[None, :]) |
                         (terminal_obs > train_max[None, :])).mean(axis=0)
    ax.bar(range(len(ood_frac_per_dim)), ood_frac_per_dim)
    ax.set_xlabel('Observation Dimension')
    ax.set_ylabel('Fraction of Terminal States OOD')
    ax.set_title('Which Dimensions Go OOD Most?')
    plt.tight_layout()
    plt.savefig("ood_analysis/per_dim_ood.png", dpi=150)
    plt.close()
    print("  Saved: ood_analysis/per_dim_ood.png")

    # Plot 5: Reward trajectory for each sampled path
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, path in enumerate(paths[:20]):  # Plot first 20 trajectories
        ax.plot(path['rewards'], alpha=0.3, color='blue')
    ax.set_xlabel('Horizon Step')
    ax.set_ylabel('Reward (-embedding distance)')
    ax.set_title('Reward Curves for Sampled Trajectories (first 20)')
    plt.tight_layout()
    plt.savefig("ood_analysis/reward_curves.png", dpi=150)
    plt.close()
    print("  Saved: ood_analysis/reward_curves.png")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Score CV (noise indicator): {cv:.3f}")
    print(f"  Raw/Emb distance correlation: {correlation:.3f}")
    print(f"  OOD ratio (NN distances): {ood_ratio:.2f}x")
    print(f"  Avg dims out of training bounds: {n_ood_dims.mean():.1f}/59")
    print(f"  States with dim > 5 std from mean: {(z_scores_all.max(axis=1) > 5).mean()*100:.1f}%")
    print()

    if cv > 0.8 and ood_ratio > 2:
        print("  DIAGNOSIS: Value function is both NOISY and OUT-OF-DISTRIBUTION.")
        print("  The high variance suggests the learned embedding is not smooth in OOD regions,")
        print("  and the high NN ratio confirms states visited are far from training data.")
    elif ood_ratio > 2:
        print("  DIAGNOSIS: Systematic OOD issue.")
        print("  States visited during planning are far from training data, so the value")
        print("  function is extrapolating. Low score variance suggests consistent mis-estimation.")
    elif cv > 0.8:
        print("  DIAGNOSIS: Value function is NOISY in-distribution.")
        print("  States are near training data but scores are highly variable, suggesting")
        print("  the embedding hasn't converged or the reward signal is too weak.")
    else:
        print("  DIAGNOSIS: Relatively in-distribution with moderate noise.")
        print("  The issue may be in the planning (horizon, temperature, action sampling)")
        print("  rather than the value function itself.")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
