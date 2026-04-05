import numpy as np
import torch
import minari
from PIL import Image



def _frames_to_tensor(frame_buffer, device):
    """Convert a list of PIL Images to a batched tensor [1, 3*frame_stack, H, W]."""
    return torch.cat(
        [torch.from_numpy(np.array(f)).float().permute(2, 0, 1) for f in frame_buffer],
        dim=0,
    ).unsqueeze(0).to(device)


def _encode_frames(frame_buffer, value_model, device):
    """Encode a full frame buffer (exactly frame_stack frames) through the frozen VIP encoder."""
    tensor = _frames_to_tensor(frame_buffer, device)
    with torch.no_grad():
        return value_model(tensor)


def _d4rl_render_84(env):
    """Render from d4rl kitchen at 84x84, pixel-identical to training images."""
    from dm_control.mujoco import engine as dm_engine
    if not hasattr(env, '_vip_camera'):
        physics = env.sim if hasattr(env, 'sim') else env.env.sim
        cam = dm_engine.MovableCamera(physics, height=1920, width=2560)
        cam.set_pose(distance=2.2, lookat=[-0.2, 0.5, 2.0], azimuth=70, elevation=-35)
        env._vip_camera = cam
    img = np.array(env._vip_camera.render())
    return Image.fromarray(img.astype(np.uint8)).resize((84, 84))


def _d4rl_set_qpos(env, qpos):
    """Set qpos in the d4rl kitchen via dm_control Physics."""
    state = env.sim.get_state()
    state[:len(qpos)] = qpos
    env.sim.set_state(state)
    env.sim.forward()


# Hardcoded from d4rl kitchen-complete-v0 episode 0 (image eval only)
SUBTASK_BOUNDARIES = [
    {'subtask_idx': 0, 'subtask_name': 'microwave',     'init_t': 0,   'goal_t': 42,  'max_steps': 84,  'baseline_tasks': 0},
    {'subtask_idx': 1, 'subtask_name': 'kettle',        'init_t': 42,  'goal_t': 79,  'max_steps': 80,  'baseline_tasks': 1},
    {'subtask_idx': 2, 'subtask_name': 'light switch',  'init_t': 79,  'goal_t': 139, 'max_steps': 120, 'baseline_tasks': 2},
    {'subtask_idx': 3, 'subtask_name': 'slide cabinet', 'init_t': 139, 'goal_t': 172, 'max_steps': 80,  'baseline_tasks': 3},
]

_SUBTASK_NAMES = ['microwave', 'kettle', 'light switch', 'slide cabinet']


def get_subtask_configs(dataset_name="D4RL/kitchen/complete-v2"):
    """Find subtask boundaries by scanning rewards in Minari episode 0 (state eval only)."""
    ep = list(minari.load_dataset(dataset_name).iterate_episodes())[0]
    obs = ep.observations["observation"]
    rewards = ep.rewards

    # Rewards are cumulative task count (0,0,...,1,1,...,2,2,...,3,3,...,4,4)
    assert rewards[-1] >= 4, f"Episode 0 only completed {int(rewards[-1])} tasks, expected 4"
    completion_times = [int(np.searchsorted(rewards, k)) for k in [1, 2, 3, 4]]

    configs = []
    for i, ct in enumerate(completion_times):
        init_t = 0 if i == 0 else completion_times[i - 1] + 1
        goal_t = ct + 1  # obs after the completing action
        duration = goal_t - init_t
        configs.append({
            'subtask_idx': i,
            'subtask_name': _SUBTASK_NAMES[i],
            'init_t': init_t,
            'goal_t': goal_t,
            'max_steps': duration * 2,
            'baseline_tasks': i,
            'goal_obs': obs[goal_t].astype(np.float32),
            'init_obs': obs[init_t].astype(np.float32),
        })
    return configs



def rollout_image_d4rl(env, policy, value_model, init_qpos_seq, goal_frames,
                       max_steps=280, device="cuda"):
    """Image rollout using d4rl kitchen renderer.

    init_qpos_seq: list of frame_stack qpos arrays (oldest to newest) for initial frame buffer.
    goal_frames: list of frame_stack PIL Images (oldest to newest) from disk at goal_t.
    value_model: frozen VIP encoder, or None if policy has its own encoder (trainable_encoder mode).
    Returns dict with total_reward, tasks_completed, episode_length.
    """
    if value_model is not None:
        obs_g = _encode_frames(goal_frames, value_model, device)
    else:
        obs_g = _frames_to_tensor(goal_frames, device)

    # Render initial frame buffer by setting qpos for each history timestep
    env.reset()
    frame_buffer = []
    for qpos in init_qpos_seq:
        _d4rl_set_qpos(env, qpos)
        frame_buffer.append(_d4rl_render_84(env))
    total_reward = 0.0
    done = False

    for step in range(max_steps):
        if value_model is not None:
            obs_s = _encode_frames(frame_buffer, value_model, device)
        else:
            obs_s = _frames_to_tensor(frame_buffer, device)
        action = policy.act(obs_s, obs_g, deterministic=True).squeeze(0)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        _, reward, done, info = env.step(action)
        total_reward += reward

        cur_frame = _d4rl_render_84(env)
        frame_buffer.pop(0)
        frame_buffer.append(cur_frame)

        if done:
            break

    return {
        "total_reward": total_reward,
        "tasks_completed": int(total_reward),
        "episode_length": step + 1,
    }


def rollout(env, policy, goal_obs, init_obs, max_steps=280, device="cuda"):
    """Execute one state-based episode and return results dict."""
    qpos = np.concatenate([init_obs[:9], init_obs[18:39]])
    qvel = np.concatenate([init_obs[9:18], init_obs[39:]])
    env.reset()
    env.unwrapped.robot_env.set_state(qpos, qvel)

    robot_obs = env.unwrapped.robot_env._get_obs()
    obs = env.unwrapped._get_obs(robot_obs)["observation"].astype(np.float32)

    goal_t = torch.tensor(goal_obs, dtype=torch.float32).unsqueeze(0).to(device)

    total_reward = 0.0
    actions_taken = []

    for step in range(max_steps):
        s_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action = policy.act(s_t, goal_t, deterministic=True).squeeze(0)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs_dict, reward, terminated, truncated, info = env.step(action)
        obs = obs_dict["observation"].astype(np.float32)
        total_reward += reward
        actions_taken.append(action.copy())

        if terminated or truncated:
            break

    tasks_completed = len(info.get("episode_task_completions", []))

    return {
        "total_reward": total_reward,
        "tasks_completed": tasks_completed,
        "episode_length": step + 1,
        "actions": actions_taken,
        "info": info,
    }


def evaluate_policy(env, policy, goal_obs, init_obs, n_episodes=10, max_steps=280, device="cuda"):
    """Run multiple rollouts and return aggregated results."""
    results = []
    for _ in range(n_episodes):
        results.append(rollout(env, policy, goal_obs, init_obs, max_steps, device))

    rewards = [r["total_reward"] for r in results]
    lengths = [r["episode_length"] for r in results]
    tasks = [r["tasks_completed"] for r in results]

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_tasks_completed": float(np.mean(tasks)),
        "std_tasks_completed": float(np.std(tasks)),
        "mean_length": float(np.mean(lengths)),
        "per_episode_tasks": tasks,
    }
