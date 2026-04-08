import numpy as np
import torch
import minari
from PIL import Image
from d4rl.kitchen.kitchen_envs import OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS, BONUS_THRESH


KITCHEN_TASKS = ['microwave', 'kettle', 'light switch', 'slide cabinet']


def detect_task_completion_times_minari(episode_obs):
    """Scan Minari episode observations to find first completion timestep per task.

    Args:
        episode_obs: the full ep.observations dict with keys
                     'observation', 'achieved_goal', 'desired_goal'.

    Returns:
        dict mapping task_name -> first completion timestep (obs index), or None.
    """
    achieved = episode_obs['achieved_goal']
    desired = episode_obs['desired_goal']
    completion = {t: None for t in KITCHEN_TASKS}
    n_obs = len(achieved[KITCHEN_TASKS[0]])

    for t in range(n_obs):
        for task in KITCHEN_TASKS:
            if completion[task] is not None:
                continue
            dist = np.linalg.norm(achieved[task][t] - desired[task][t])
            if dist < BONUS_THRESH:
                completion[task] = t
    return completion


def detect_task_completion_times_d4rl(observations):
    """Scan d4rl 60D observations to find first completion timestep per task.

    Args:
        observations: (T, 60) array of d4rl observations.

    Returns:
        dict mapping task_name -> first completion timestep (obs index), or None.
    """
    completion = {t: None for t in KITCHEN_TASKS}

    for t in range(len(observations)):
        qpos = observations[t, 0:30]
        for task in KITCHEN_TASKS:
            if completion[task] is not None:
                continue
            idx = OBS_ELEMENT_INDICES[task]
            dist = np.linalg.norm(qpos[idx] - OBS_ELEMENT_GOALS[task])
            if dist < BONUS_THRESH:
                completion[task] = t
    return completion


def get_subtask_range(completions, task_name):
    """Given task completion times, return (init_t, goal_t) for a specific task.

    init_t: first timestep of this subtask (after previous task completed, or 0).
    goal_t: first timestep where the task is complete (goal observation index).
    """
    ct = completions[task_name]

    # init_t = after the latest task that was completed before this one
    prev_ct = -1
    for other_ct in completions.values():
        if other_ct < ct:
            prev_ct = max(prev_ct, other_ct)

    init_t = prev_ct + 1  # 0 if this task was completed first
    return (init_t, ct)


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



def get_full_sequence_config(dataset_name="D4RL/kitchen/complete-v2"):
    """Return init/goal configs for full-sequence eval from all episodes."""
    dataset = minari.load_dataset(dataset_name, download=True)
    eval_states = []
    for ep in dataset.iterate_episodes():
        obs = ep.observations["observation"]
        eval_states.append({
            'init_obs': obs[0].astype(np.float32),
            'goal_obs': obs[-1].astype(np.float32),
            'max_steps': len(obs) * 2,
        })
    return eval_states


def get_subtask_configs(dataset_name="D4RL/kitchen/complete-v2"):
    """Extract subtask eval configs from all kitchen-complete episodes.

    Returns dict mapping task_name -> list of eval dicts with init/goal
    observations and subtask boundaries.
    """
    dataset = minari.load_dataset(dataset_name, download=True)
    configs = {t: [] for t in KITCHEN_TASKS}

    for ep in dataset.iterate_episodes():
        completions = detect_task_completion_times_minari(ep.observations)
        obs = ep.observations['observation']

        for task_name in KITCHEN_TASKS:
            init_t, goal_t = get_subtask_range(completions, task_name)
            duration = goal_t - init_t
            baseline = sum(1 for ct in completions.values() if ct < init_t)
            configs[task_name].append({
                'subtask_name': task_name,
                'init_t': init_t,
                'goal_t': goal_t,
                'max_steps': duration * 2,
                'baseline_tasks': baseline,
                'goal_obs': obs[goal_t].astype(np.float32),
                'init_obs': obs[init_t].astype(np.float32),
            })
    return configs


def get_image_subtask_configs(datapath, env_name, frame_stack=1):
    """Extract subtask image eval configs from all d4rl kitchen-complete trajectories.

    Returns dict mapping task_name -> list of eval dicts with init qpos,
    goal frames, and subtask boundaries.
    """
    import os
    import glob
    import re
    import gym
    import d4rl

    env = gym.make(env_name)
    dataset = env.get_dataset()
    all_obs = dataset['observations']
    terminals = np.where(dataset['terminals'])[0]
    env.close()

    traj_dirs = sorted(glob.glob(os.path.join(datapath, "traj*")),
                       key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

    configs = {t: [] for t in KITCHEN_TASKS}
    obs_id = 0
    for traj_id, term_idx in enumerate(terminals):
        if traj_id >= len(traj_dirs):
            break
        traj_len = term_idx - obs_id + 1
        traj_obs = all_obs[obs_id:obs_id + traj_len]
        traj_dir = traj_dirs[traj_id]
        completions = detect_task_completion_times_d4rl(traj_obs)

        for task_name in KITCHEN_TASKS:
            init_t, goal_t = get_subtask_range(completions, task_name)
            duration = goal_t - init_t
            baseline = sum(1 for ct in completions.values() if ct < init_t)
            goal_frames = [
                Image.open(os.path.join(traj_dir, f"img{max(0, goal_t - frame_stack + k + 1)}.png"))
                for k in range(frame_stack)
            ]
            init_qpos_seq = [all_obs[obs_id + max(0, init_t - frame_stack + k + 1), :30]
                             for k in range(frame_stack)]
            configs[task_name].append({
                'init_qpos_seq': init_qpos_seq,
                'goal_frames': goal_frames,
                'max_steps': duration * 2,
                'baseline_tasks': baseline,
            })

        obs_id = term_idx + 1

    return configs


def get_image_full_sequence_configs(datapath, env_name, frame_stack=1):
    """Collect full-sequence image eval states from all d4rl trajectories."""
    import os
    import glob
    import re
    import gym
    import d4rl

    env = gym.make(env_name)
    dataset = env.get_dataset()
    all_obs = dataset['observations']
    terminals = np.where(dataset['terminals'])[0]
    env.close()

    traj_dirs = sorted(glob.glob(os.path.join(datapath, "traj*")),
                       key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

    eval_states = []
    obs_id = 0
    for traj_id, term_idx in enumerate(terminals):
        if traj_id >= len(traj_dirs):
            break
        traj_len = term_idx - obs_id + 1
        traj_dir = traj_dirs[traj_id]

        goal_frames = [
            Image.open(os.path.join(traj_dir, f"img{max(0, traj_len - 1 - frame_stack + k + 1)}.png"))
            for k in range(frame_stack)
        ]
        init_qpos_seq = [all_obs[obs_id, :30] for _ in range(frame_stack)]
        eval_states.append({
            'init_qpos_seq': init_qpos_seq,
            'goal_frames': goal_frames,
            'max_steps': traj_len * 2,
        })

        obs_id = term_idx + 1

    return eval_states


def rollout_image_d4rl(env, policy, value_model, init_qpos_seq, goal_frames,
                       max_steps=280, device="cuda"):
    """Image rollout using d4rl kitchen renderer.

    init_qpos_seq: list of frame_stack qpos arrays (oldest to newest) for initial frame buffer.
    goal_frames: list of frame_stack PIL Images (oldest to newest) from disk at goal_t.
    value_model: frozen VIP encoder, or None if policy has its own encoder (use_pretrained mode).
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


def evaluate_policy(env, policy, eval_states, max_steps=280, device="cuda"):
    """Run one deterministic rollout per eval state and return aggregated results.

    eval_states: list of dicts with 'init_obs', 'goal_obs' (and optionally 'max_steps').
    """
    results = []
    for es in eval_states:
        ms = es.get('max_steps', max_steps)
        results.append(rollout(env, policy, es['goal_obs'], es['init_obs'], ms, device))

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
