import os
import re
import glob
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


class PolicyExtractionBuffer(Dataset):
    """Finite dataset of (s, a, s_next, g) transitions from Minari Kitchen data.

    Goals are sampled as random future states in the same episode,
    matching the VIP/IQL convention.

    When subtask_name is set, only transitions relevant to that subtask are
    included and goals are constrained to the subtask's completion window.
    """

    def __init__(self, datasource="D4RL/kitchen/complete-v2", num_goal_transitions=3,
                 seed=42, subtask_name=None, noise_ratio=0.0, noise_dir=""):
        import minari
        from policy_extraction.evaluate import (
            detect_task_completion_times_minari, get_subtask_range,
        )

        rng = np.random.RandomState(seed)
        dataset = minari.load_dataset(datasource, download=True)

        # Collect (s, a, s', g) transitions
        transitions = []
        for ep in dataset.iterate_episodes():
            obs = ep.observations['observation']  # (T+1, obs_dim)
            acts = ep.actions                      # (T, action_dim)

            if subtask_name is not None:
                completions = detect_task_completion_times_minari(ep.observations)
                init_t, goal_t = get_subtask_range(completions, subtask_name)
                goal_obs = obs[goal_t].astype(np.float32)
                for t in range(init_t, goal_t):
                    transitions.append((obs[t], acts[t], obs[t + 1], goal_obs))
            else:
                for t in range(len(acts)):
                    for _ in range(num_goal_transitions):
                        t_g = rng.randint(t + 1, len(obs))
                        transitions.append((obs[t], acts[t], obs[t + 1], obs[t_g]))

        # Mix in noisy transitions from pre-generated data on disk
        if subtask_name is not None and noise_ratio > 0 and noise_dir:
            n_noise = int(len(transitions) * noise_ratio)
            n_demo = len(transitions) - n_noise

            ds_short = datasource.split("/")[-1]  # e.g. "complete-v2"
            subtask_key = subtask_name.replace(" ", "_")
            noise_path = os.path.join(noise_dir, "state",
                                      f"{ds_short}_{subtask_key}_seed42.pt")
            noise_data = torch.load(noise_path, map_location="cpu", weights_only=False)
            noise_idx = rng.choice(len(noise_data["cur"]), n_noise, replace=False)
            noise = [
                (np.asarray(noise_data["cur"][i]), np.asarray(noise_data["action"][i]),
                 np.asarray(noise_data["nxt"][i]), np.asarray(noise_data["goal"][i]))
                for i in noise_idx
            ]

            demo_idx = rng.choice(len(transitions), n_demo, replace=False)
            transitions = [transitions[i] for i in demo_idx] + noise
            print(f"  Loaded {len(noise)} noise from {noise_path}")
            print(f"  {n_demo} demo + {len(noise)} noise (ratio={noise_ratio})")

        # Convert to arrays
        states, actions, next_states, goals = zip(*transitions)
        self.states      = np.stack(states).astype(np.float32)
        self.actions     = np.stack(actions).astype(np.float32)
        self.next_states = np.stack(next_states).astype(np.float32)
        self.goals       = np.stack(goals).astype(np.float32)

        label = f"subtask={subtask_name}" if subtask_name else "full-sequence"
        print(f"PolicyExtractionBuffer ({label}): {len(self)} transitions from {datasource}")
        if len(self) > 0:
            print(f"  obs_dim={self.states.shape[1]}, action_dim={self.actions.shape[1]}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.goals[idx]),
        )


def _sort_traj_dirs(dirs):
    """Sort traj directories numerically (traj0, traj1, ..., traj10, not traj0, traj1, traj10)."""
    return sorted(dirs, key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))


class ImagePolicyExtractionBuffer(Dataset):
    """Dataset of (image_t, action_t, image_{t+1}, image_goal) from paired image + d4rl data.

    Images come from disk (traj*/img*.png). Actions come from the d4rl
    kitchen dataset using the same flat indexing as the original
    image generation script (vis_franka_kitchen.py).

    When subtask_name is set, only transitions relevant to that subtask are
    included and goals are constrained to the subtask's completion window.
    """

    def __init__(self, datapath, num_goal_transitions=3, seed=42, frame_stack=1,
                 subtask_name=None, noise_ratio=0.0, noise_dir=""):
        import gym
        import d4rl
        from policy_extraction.evaluate import (
            detect_task_completion_times_d4rl, get_subtask_range,
        )

        rng = np.random.RandomState(seed)
        self.frame_stack = frame_stack

        env_name = os.path.basename(datapath.rstrip("/"))
        env = gym.make(env_name)
        dataset = env.get_dataset()
        all_actions = dataset["actions"]
        all_obs = dataset["observations"]
        terminals = np.where(dataset["terminals"])[0]
        env.close()

        traj_dirs = _sort_traj_dirs(glob.glob(os.path.join(datapath, "traj*")))

        # Collect (img_t, action, img_t1, img_g) transitions
        transitions = []
        obs_id = 0
        for traj_id, term_idx in enumerate(terminals):
            if traj_id >= len(traj_dirs):
                break
            traj_len = term_idx - obs_id + 1
            traj_dir = traj_dirs[traj_id]

            if subtask_name is not None:
                traj_obs = all_obs[obs_id:obs_id + traj_len]
                completions = detect_task_completion_times_d4rl(traj_obs)
                init_t, goal_t = get_subtask_range(completions, subtask_name)
                img_g = [os.path.join(traj_dir, f"img{max(0, goal_t - frame_stack + k + 1)}.png")
                         for k in range(frame_stack)]
                for t in range(init_t, goal_t):
                    action = all_actions[obs_id + t].astype(np.float32)
                    img_t  = [os.path.join(traj_dir, f"img{max(0, t - frame_stack + k + 1)}.png")
                              for k in range(frame_stack)]
                    img_t1 = [os.path.join(traj_dir, f"img{max(0, t - frame_stack + k + 2)}.png")
                              for k in range(frame_stack)]
                    transitions.append((img_t, action, img_t1, img_g))
            else:
                for t in range(traj_len - 1):
                    action = all_actions[obs_id + t].astype(np.float32)
                    img_t  = [os.path.join(traj_dir, f"img{max(0, t - frame_stack + k + 1)}.png")
                              for k in range(frame_stack)]
                    img_t1 = [os.path.join(traj_dir, f"img{max(0, t - frame_stack + k + 2)}.png")
                              for k in range(frame_stack)]
                    for _ in range(num_goal_transitions):
                        t_g = rng.randint(t + 1, traj_len)
                        img_g = [os.path.join(traj_dir, f"img{max(0, t_g - frame_stack + k + 1)}.png")
                                 for k in range(frame_stack)]
                        transitions.append((img_t, action, img_t1, img_g))

            obs_id = term_idx + 1

        # Mix in noisy transitions from pre-generated data on disk
        if subtask_name is not None and noise_ratio > 0 and noise_dir:
            n_noise = int(len(transitions) * noise_ratio)
            n_demo = len(transitions) - n_noise

            subtask_key = subtask_name.replace(" ", "_")
            noise_subdir = os.path.join(noise_dir, "image",
                                        f"{env_name}_{subtask_key}_fs{frame_stack}_seed42")
            noise_actions = torch.load(os.path.join(noise_subdir, "actions.pt"),
                                       map_location="cpu", weights_only=False)
            if isinstance(noise_actions, torch.Tensor):
                noise_actions = noise_actions.numpy()

            noise_idx = rng.choice(len(noise_actions), n_noise, replace=False)
            noise = []
            for i in noise_idx:
                if frame_stack == 1:
                    cur_paths = [os.path.join(noise_subdir, f"cur_{i}.png")]
                    nxt_paths = [os.path.join(noise_subdir, f"nxt_{i}.png")]
                    goal_paths = [os.path.join(noise_subdir, f"goal_{i}.png")]
                else:
                    cur_paths = [os.path.join(noise_subdir, f"cur_{i}_f{k}.png")
                                 for k in range(frame_stack)]
                    nxt_paths = [os.path.join(noise_subdir, f"nxt_{i}_f{k}.png")
                                 for k in range(frame_stack)]
                    goal_paths = [os.path.join(noise_subdir, f"goal_{i}_f{k}.png")
                                  for k in range(frame_stack)]
                action = noise_actions[i].astype(np.float32)
                noise.append((cur_paths, action, nxt_paths, goal_paths))

            demo_idx = rng.choice(len(transitions), n_demo, replace=False)
            transitions = [transitions[i] for i in demo_idx] + noise
            print(f"  Loaded {len(noise)} noise from {noise_subdir}")
            print(f"  {n_demo} demo + {len(noise)} noise (ratio={noise_ratio})")

        self.transitions = transitions
        label = f"subtask={subtask_name}" if subtask_name else "full-sequence"
        print(f"ImagePolicyExtractionBuffer ({label}): {len(self)} transitions from "
              f"{len(traj_dirs)} traj(s) (frame_stack={frame_stack})")

    def __len__(self):
        return len(self.transitions)

    def _load_frames(self, frames):
        """Load frames from paths (list of str) or return pre-loaded tensor."""
        if isinstance(frames, torch.Tensor):
            return frames
        return torch.cat([torchvision.io.read_image(p).float() for p in frames], dim=0)

    def __getitem__(self, idx):
        img_t_src, action, img_t1_src, img_g_src = self.transitions[idx]
        img_t  = self._load_frames(img_t_src)
        img_t1 = self._load_frames(img_t1_src)
        img_g  = self._load_frames(img_g_src)
        return img_t, torch.from_numpy(action), img_t1, img_g
