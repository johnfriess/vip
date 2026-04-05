import os
import re
import glob
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

# Kitchen datasets that get combined when using "all-v2"
KITCHEN_COMBINED_SOURCES = [
    "D4RL/kitchen/complete-v2",
    "D4RL/kitchen/partial-v2",
    "D4RL/kitchen/mixed-v2",
]

# complete + mixed only (no partial)
KITCHEN_COMPLETE_MIXED_SOURCES = [
    "D4RL/kitchen/complete-v2",
    "D4RL/kitchen/mixed-v2",
]

# Image datasets that get combined when datapath="kitchen-image-all"
KITCHEN_IMAGE_ALL = "kitchen-image-all"
KITCHEN_IMAGE_COMPLETE_DATAPATH = "/data/siddhant/vip/vis_data/kitchen-complete-v0"
KITCHEN_IMAGE_COMBINED_DATAPATHS = [
    KITCHEN_IMAGE_COMPLETE_DATAPATH,
    "/data/siddhant/vip/vis_data/kitchen-mixed-v0",
]


class PolicyExtractionBuffer(Dataset):
    """Finite dataset of (s, a, s_next, g) tuples from Minari Kitchen data.

    Goals are sampled as random future states in the same episode,
    matching the VIP/IQL convention.
    """

    def __init__(self, datasource="D4RL/kitchen/complete-v2", num_goal_samples=3, seed=42):
        import minari

        rng = np.random.RandomState(seed)
        if datasource == "D4RL/kitchen/all-v2":
            sources = KITCHEN_COMBINED_SOURCES
        elif datasource == "D4RL/kitchen/complete-mixed-v2":
            sources = KITCHEN_COMPLETE_MIXED_SOURCES
        else:
            sources = [datasource]

        self.states = []
        self.actions = []
        self.next_states = []
        self.goals = []

        for source in sources:
            dataset = minari.load_dataset(source, download=True)
            for ep in dataset.iterate_episodes():
                obs = ep.observations['observation']  # (T+1, obs_dim)
                acts = ep.actions                      # (T, action_dim)
                episode_len = len(obs) - 1             # T transitions

                if episode_len < 2:
                    continue

                for t in range(episode_len):
                    for _ in range(num_goal_samples):
                        t_g = rng.randint(t + 1, len(obs))
                        self.states.append(obs[t].astype(np.float32))
                        self.actions.append(acts[t].astype(np.float32))
                        self.next_states.append(obs[t + 1].astype(np.float32))
                        self.goals.append(obs[t_g].astype(np.float32))

        self.states = np.array(self.states)
        self.actions = np.array(self.actions)
        self.next_states = np.array(self.next_states)
        self.goals = np.array(self.goals)

        print(f"PolicyExtractionBuffer: {len(self)} transitions from {len(sources)} source(s)")
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
    kitchen-complete-v0 dataset using the same flat indexing as the original
    image generation script (vis_franka_kitchen.py).
    """

    def __init__(self, datapath, num_goal_samples=3, seed=42, frame_stack=1):
        import gym
        import d4rl

        rng = np.random.RandomState(seed)
        self.frame_stack = frame_stack

        datapaths = KITCHEN_IMAGE_COMBINED_DATAPATHS if datapath == KITCHEN_IMAGE_ALL else [datapath]

        self.samples = []
        self.weights = None  # set externally for trainable_encoder mode
        total_trajs = 0

        for dp in datapaths:
            env_name = os.path.basename(dp.rstrip("/"))
            env = gym.make(env_name)
            dataset = env.get_dataset()
            all_actions = dataset["actions"]
            # Use only terminals (not timeouts) to match vis_franka_kitchen.py,
            # which only splits trajectories on terminal flags.
            terminals = np.where(dataset["terminals"])[0]
            env.close()

            traj_dirs = _sort_traj_dirs(glob.glob(os.path.join(dp, "traj*")))

            obs_id = 0
            for traj_id, term_idx in enumerate(terminals):
                if traj_id >= len(traj_dirs):
                    break
                traj_len = term_idx - obs_id + 1
                traj_dir = traj_dirs[traj_id]

                # Sliding window: state=[f_{t-fs+1}..f_t], action=a_t, next=[f_{t-fs+2}..f_{t+1}]
                for t in range(traj_len - 1):
                    action = all_actions[obs_id + t].astype(np.float32)
                    img_t  = [os.path.join(traj_dir, f"img{max(0, t - frame_stack + k + 1)}.png")
                              for k in range(frame_stack)]
                    img_t1 = [os.path.join(traj_dir, f"img{max(0, t - frame_stack + k + 2)}.png")
                              for k in range(frame_stack)]
                    for _ in range(num_goal_samples):
                        t_g  = rng.randint(t + 1, traj_len)
                        img_g = [os.path.join(traj_dir, f"img{max(0, t_g - frame_stack + k + 1)}.png")
                                 for k in range(frame_stack)]
                        self.samples.append((img_t, action, img_t1, img_g))

                obs_id = term_idx + 1

            total_trajs += len(traj_dirs)

        print(f"ImagePolicyExtractionBuffer: {len(self)} transitions from {total_trajs} traj(s) "
              f"across {len(datapaths)} dataset(s) (frame_stack={frame_stack})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_paths_t, action, img_paths_t1, img_paths_g = self.samples[idx]
        img_t  = torch.cat([torchvision.io.read_image(p).float() for p in img_paths_t],  dim=0)
        img_t1 = torch.cat([torchvision.io.read_image(p).float() for p in img_paths_t1], dim=0)
        img_g  = torch.cat([torchvision.io.read_image(p).float() for p in img_paths_g],  dim=0)
        if self.weights is not None:
            return img_t, torch.from_numpy(action), img_t1, img_g, self.weights[idx]
        return img_t, torch.from_numpy(action), img_t1, img_g
