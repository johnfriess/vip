# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import warnings

import torchvision
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import glob
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
import pandas as pd
import json
import time
import pickle
from torchvision.utils import save_image
import json
import random
import h5py

STATE_DATASETS = [
    "D4RL/kitchen/complete-v2",
    "D4RL/kitchen/partial-v2",
    "D4RL/kitchen/mixed-v2",
    "D4RL/kitchen/all-v2",
    "robomimic"
]

KITCHEN_COMBINED_SOURCES = [
    "D4RL/kitchen/complete-v2",
    "D4RL/kitchen/partial-v2",
    "D4RL/kitchen/mixed-v2",
]

KITCHEN_IMAGE_ALL = "kitchen-image-all"
KITCHEN_IMAGE_COMBINED_DATAPATHS = [
    "/data/siddhant/vip/vis_data/kitchen-complete-v0",
    "/data/siddhant/vip/vis_data/kitchen-mixed-v0",
]

# Default observation keys for robomimic low-dim datasets
ROBOMIMIC_DEFAULT_OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
]

def get_ind(vid, index, ds="ego4d"):
    if ds == "ego4d":
        return torchvision.io.read_image(f"{vid}{index:06}.jpg")
    elif ds in ("kitchen-image", KITCHEN_IMAGE_ALL):
        return torchvision.io.read_image(f"{vid}/img{index}.png")
    else:
        try:
            return torchvision.io.read_image(f"{vid}/{index}.jpg")
        except:
            return torchvision.io.read_image(f"{vid}/{index}.png")

## Data Loader for VIP
class VIPBuffer(IterableDataset):
    def __init__(self, datasource='ego4d', datapath=None, num_workers=10, doaug="none", frame_stack=1, frame_combine=False):
        self._num_workers = max(1, num_workers)
        self.datasource = datasource
        self.datapath = datapath
        assert datapath is not None or datasource == KITCHEN_IMAGE_ALL
        self.doaug = doaug
        self.frame_stack = frame_stack
        self.frame_combine = frame_combine  # non-overlapping pairs: obs[i] = [frame[i*fs], ..., frame[i*fs+fs-1]]

        # Augmentations
        self.preprocess = torch.nn.Sequential(
                        transforms.Resize(256),
                        transforms.CenterCrop(224)
                )
        if doaug in ["rc", "rctraj"]:
            self.aug = torch.nn.Sequential(
                transforms.RandomResizedCrop(224, scale = (0.2, 1.0)),
            )
        else:
            self.aug = lambda a : a

        # Load Data
        if "ego4d" == self.datasource:
            print("Ego4D")
            self.manifest = pd.read_csv(f"{self.datapath}/manifest.csv")
            print(self.manifest)
            self.ego4dlen = len(self.manifest)
        elif self.datasource == "kitchen-image":
            assert datapath is not None, "datapath required for kitchen-image"
            self.episodes = []
            for episode in sorted(glob.glob(f"{datapath}/traj*")):
                episode_len = len(glob.glob(f"{episode}/img*.png"))
                if episode_len > frame_stack + 1:
                    self.episodes.append((episode, episode_len))
        elif self.datasource == KITCHEN_IMAGE_ALL:
            self.episodes = []
            for dp in KITCHEN_IMAGE_COMBINED_DATAPATHS:
                for episode in sorted(glob.glob(f"{dp}/traj*")):
                    episode_len = len(glob.glob(f"{episode}/img*.png"))
                    if episode_len > frame_stack + 1:
                        self.episodes.append((episode, episode_len))
            print(f"kitchen-image-all: loaded {len(self.episodes)} episodes from {len(KITCHEN_IMAGE_COMBINED_DATAPATHS)} datasets")

    def _load_stacked_obs(self, vid, idx):
        frames = []
        if self.frame_combine:
            # Non-overlapping: obs[i] = [frame[i*fs + k] for k in range(fs)]
            for k in range(self.frame_stack):
                frames.append(get_ind(vid, idx * self.frame_stack + k, self.datasource))
        else:
            # Sliding window: obs[i] = [frame[i-fs+1], ..., frame[i]]
            for k in range(self.frame_stack):
                frame_idx = max(0, idx - self.frame_stack + k + 1)
                frames.append(get_ind(vid, frame_idx, self.datasource))
        return torch.cat(frames, dim=0)

    def _effective_vidlen(self, vidlen):
        """Returns the number of observations given raw frame count."""
        if self.frame_combine and self.frame_stack > 1:
            return vidlen // self.frame_stack
        return vidlen

    def _get_vid_and_len(self):
        if self.datasource == 'ego4d':
            vidid = np.random.randint(0, self.ego4dlen)
            m = self.manifest.iloc[vidid]
            return m["path"], m["len"]
        elif self.datasource in ('kitchen-image', KITCHEN_IMAGE_ALL):
            episode_ind = np.random.randint(0, len(self.episodes))
            return self.episodes[episode_ind]
        else:
            video_paths = glob.glob(f"{self.datapath}/[0-9]*")
            video_id = np.random.randint(0, len(video_paths))
            vid = video_paths[video_id]
            vidlen = len(glob.glob(f'{vid}/*.png'))
            if vidlen == 0:
                vidlen = len(glob.glob(f'{vid}/*.jpg'))
            return vid, vidlen

    def get_trajectory(self):
        vid, vidlen = self._get_vid_and_len()
        eff_len = self._effective_vidlen(vidlen)
        frames = []
        for i in range(eff_len):
            if self.frame_stack > 1:
                frames.append(self._load_stacked_obs(vid, i))
            else:
                frames.append(get_ind(vid, i, self.datasource))
        return torch.stack(frames).float()

    def _sample(self):
        vid, vidlen = self._get_vid_and_len()
        vidlen = self._effective_vidlen(vidlen)

        # Sample (o_t, o_k, o_k+1, o_T) for VIP training
        start_ind = np.random.randint(0, vidlen-2)
        end_ind = np.random.randint(start_ind+1, vidlen)

        s0_ind_vip = np.random.randint(start_ind, end_ind)
        s1_ind_vip = min(s0_ind_vip+1, end_ind)

        # Self-supervised reward (this is always -1)
        reward = float(s0_ind_vip == end_ind) - 1

        # Load frames (with frame stacking if frame_stack > 1)
        if self.frame_stack > 1:
            im0 = self._load_stacked_obs(vid, start_ind)
            img = self._load_stacked_obs(vid, end_ind)
            imts0_vip = self._load_stacked_obs(vid, s0_ind_vip)
            imts1_vip = self._load_stacked_obs(vid, s1_ind_vip)
        else:
            im0 = get_ind(vid, start_ind, self.datasource)
            img = get_ind(vid, end_ind, self.datasource)
            imts0_vip = get_ind(vid, s0_ind_vip, self.datasource)
            imts1_vip = get_ind(vid, s1_ind_vip, self.datasource)

        # Apply augmentation
        if self.doaug == "rctraj":
            ### Encode each image in the video at once the same way
            allims = torch.stack([im0, img, imts0_vip, imts1_vip], 0)
            allims_aug = self.aug(allims / 255.0) * 255.0
            im0 = allims_aug[0]
            img = allims_aug[1]
            imts0_vip = allims_aug[2]
            imts1_vip = allims_aug[3]
        else:
            ### Encode each image individually
            im0 = self.aug(im0 / 255.0) * 255.0
            img = self.aug(img / 255.0) * 255.0
            imts0_vip = self.aug(imts0_vip / 255.0) * 255.0
            imts1_vip = self.aug(imts1_vip / 255.0) * 255.0

        im = torch.stack([im0, img, imts0_vip, imts1_vip])
        im = self.preprocess(im)
        return (im, reward)

    def __iter__(self):
        while True:
            yield self._sample()

class StateVIPBuffer(IterableDataset):
    def __init__(self, datasource="D4RL/kitchen/complete-v2", use_achieved_goal=False):
        import minari

        self.use_achieved_goal = use_achieved_goal

        # Determine which datasets to load
        if datasource == "D4RL/kitchen/all-v2":
            sources = KITCHEN_COMBINED_SOURCES
        else:
            sources = [datasource]

        self.episodes = []
        for source in sources:
            dataset = minari.load_dataset(source, download=True)
            for ep in dataset.iterate_episodes():
                if len(ep.observations['observation']) > 2:
                    if use_achieved_goal:
                        achieved = ep.observations['achieved_goal']
                        # Flatten achieved_goal: kettle(7) + light(2) + microwave(1) + slide(1) = 11D
                        flattened = np.concatenate([
                            achieved['kettle'],
                            achieved['light switch'],
                            achieved['microwave'],
                            achieved['slide cabinet']
                        ], axis=1)
                        self.episodes.append(flattened)
                    else:
                        self.episodes.append(ep.observations['observation'])

    def get_trajectory(self):
        episode_ind = np.random.randint(0, len(self.episodes))
        return torch.tensor(self.episodes[episode_ind], dtype=torch.float32)

    def _sample(self):
        episode_ind = np.random.randint(0, len(self.episodes))
        obs = self.episodes[episode_ind]  # (T+1, obs_dim)
        episode_len = len(obs) - 1  # T transitions

        # Sample (o_t, o_k, o_k+1, o_T) for VIP training
        start_ind = np.random.randint(0, episode_len - 1)
        end_ind = np.random.randint(start_ind + 1, episode_len + 1)

        s0_ind_vip = np.random.randint(start_ind, end_ind)
        s1_ind_vip = min(s0_ind_vip + 1, end_ind)

        ob0 = obs[start_ind]
        obg = obs[end_ind]
        obs0_vip = obs[s0_ind_vip]
        obs1_vip = obs[s1_ind_vip]

        # Self-supervised reward (this is always -1)
        reward = float(s0_ind_vip == end_ind) - 1

        ob = torch.from_numpy(np.stack([ob0, obg, obs0_vip, obs1_vip]).astype(np.float32))
        return (ob, reward)

    def __iter__(self):
        while True:
            yield self._sample()

class StateIQLBuffer(IterableDataset):
    def __init__(self, datasource="D4RL/kitchen/complete-v2"):
        import minari

        # Determine which datasets to load
        if datasource == "D4RL/kitchen/all-v2":
            sources = KITCHEN_COMBINED_SOURCES
        else:
            sources = [datasource]

        # Store EpisodeData objects directly
        # Minari kitchen obs is 59D (goals stored separately, not embedded)
        self.episodes = []
        for source in sources:
            dataset = minari.load_dataset(source, download=True)
            for ep in dataset.iterate_episodes():
                if len(ep.observations['observation']) > 1:  # Need at least 2 states
                    self.episodes.append(ep)

    def get_trajectory(self):
        episode_ind = np.random.randint(0, len(self.episodes))
        ep = self.episodes[episode_ind]
        obs = ep.observations['observation'].astype(np.float32)  # (T+1, obs_dim)
        traj = torch.from_numpy(obs)
        g = traj[-1]  # (D,)
        g_rep = g.unsqueeze(0).expand(traj.shape[0], -1)  # (T+1, D)
        return torch.cat([traj, g_rep], dim=-1)

    def _sample(self):
        episode_ind = np.random.randint(0, len(self.episodes))
        ep = self.episodes[episode_ind]
        obs = ep.observations['observation']  # (T+1, obs_dim)
        episode_len = len(obs) - 1  # T transitions

        # Ensure there are at least 2 transitions in the episode
        if episode_len < 2:
            return self._sample()

        # Sample (o_t, o_t+1, o_T) for IQL training
        t = np.random.randint(0, episode_len - 1)
        t_g = np.random.randint(t + 1, episode_len)

        g = torch.from_numpy(obs[t_g].astype(np.float32))
        s = torch.from_numpy(obs[t].astype(np.float32))
        s_next = torch.from_numpy(obs[t + 1].astype(np.float32))
        a = torch.from_numpy(ep.actions[t].astype(np.float32))

        reached = t + 1 == t_g
        is_terminal = ep.terminations[t] or ep.truncations[t]
        discount = torch.tensor(0.0 if is_terminal or reached else 1.0, dtype=torch.float32)
        r = torch.tensor(-1.0, dtype=torch.float32)

        # Represent current state and goal state as one input state
        ob = torch.cat([s, g], dim=-1)
        ob_next = torch.cat([s_next, g], dim=-1)
        return (ob, a, r, discount, ob_next)

    def __iter__(self):
        while True:
            yield self._sample()

class RobomimicVIPBuffer(IterableDataset):
    """Data loader for robomimic HDF5 datasets for VIP training.

    Supports both low-dimensional observations and raw states.
    """
    def __init__(self, hdf5_path, obs_keys=None, filter_key=None, use_states=False):
        """
        Args:
            hdf5_path: Path to robomimic HDF5 file
            obs_keys: List of observation keys to concatenate. If None, uses defaults.
            filter_key: 'train', 'valid', or None for all demos
            use_states: If True, use raw 'states' instead of 'obs'
        """
        self.hdf5_path = hdf5_path
        self.obs_keys = obs_keys
        self.filter_key = filter_key
        self.use_states = use_states

        # Load dataset
        self.hdf5_file = h5py.File(hdf5_path, 'r')

        # Get demo keys
        if filter_key is not None and f"mask/{filter_key}" in self.hdf5_file:
            self.demo_keys = [
                key.decode('utf-8') if isinstance(key, bytes) else key
                for key in self.hdf5_file[f"mask/{filter_key}"][:]
            ]
        else:
            self.demo_keys = [
                key for key in self.hdf5_file["data"].keys()
                if key.startswith("demo_")
            ]

        # Preload all episodes for faster sampling
        self.episodes = []
        for demo_key in self.demo_keys:
            demo = self.hdf5_file[f"data/{demo_key}"]
            if self.use_states:
                obs = demo["states"][:]
            else:
                obs = self._get_obs(demo)
            if len(obs) > 2:  # Need at least 3 observations
                self.episodes.append(obs.astype(np.float32))

    def _get_obs(self, demo):
        """Concatenate observation keys into a single observation vector."""
        obs_group = demo["obs"]
        keys = self.obs_keys if self.obs_keys else ROBOMIMIC_DEFAULT_OBS_KEYS

        obs_arrays = []
        for key in keys:
            if key in obs_group:
                arr = obs_group[key][:]
                if arr.ndim > 2:
                    # Flatten if needed (e.g., images would be skipped in state-based)
                    arr = arr.reshape(arr.shape[0], -1)
                obs_arrays.append(arr)

        if not obs_arrays:
            # Fallback to states if no obs keys found
            return demo["states"][:]

        return np.concatenate(obs_arrays, axis=-1)

    def get_trajectory(self):
        """Return a random trajectory for visualization."""
        episode_ind = np.random.randint(0, len(self.episodes))
        return torch.tensor(self.episodes[episode_ind], dtype=torch.float32)

    def _sample(self):
        episode_ind = np.random.randint(0, len(self.episodes))
        obs = self.episodes[episode_ind]  # (T+1, obs_dim)
        episode_len = len(obs) - 1  # T transitions

        # Sample (o_start, o_goal, o_t, o_t+1) for VIP training
        start_ind = np.random.randint(0, episode_len - 1)
        end_ind = np.random.randint(start_ind + 1, episode_len + 1)

        s0_ind_vip = np.random.randint(start_ind, end_ind)
        s1_ind_vip = min(s0_ind_vip + 1, end_ind)

        ob0 = obs[start_ind]
        obg = obs[end_ind]
        obs0_vip = obs[s0_ind_vip]
        obs1_vip = obs[s1_ind_vip]

        # Self-supervised reward (always -1 for VIP)
        reward = float(s0_ind_vip == end_ind) - 1

        ob = torch.from_numpy(np.stack([ob0, obg, obs0_vip, obs1_vip]))
        return (ob, reward)

    def __iter__(self):
        while True:
            yield self._sample()

    def __del__(self):
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()


class RobomimicIQLBuffer(IterableDataset):
    """Data loader for robomimic HDF5 datasets for IQL training.

    Returns (state, action, reward, discount, next_state) tuples with goal conditioning.
    """
    def __init__(self, hdf5_path, obs_keys=None, filter_key=None, use_states=False):
        """
        Args:
            hdf5_path: Path to robomimic HDF5 file
            obs_keys: List of observation keys to concatenate. If None, uses defaults.
            filter_key: 'train', 'valid', or None for all demos
            use_states: If True, use raw 'states' instead of 'obs'
        """
        self.hdf5_path = hdf5_path
        self.obs_keys = obs_keys
        self.filter_key = filter_key
        self.use_states = use_states

        # Load dataset
        self.hdf5_file = h5py.File(hdf5_path, 'r')

        # Get demo keys
        if filter_key is not None and f"mask/{filter_key}" in self.hdf5_file:
            self.demo_keys = [
                key.decode('utf-8') if isinstance(key, bytes) else key
                for key in self.hdf5_file[f"mask/{filter_key}"][:]
            ]
        else:
            self.demo_keys = [
                key for key in self.hdf5_file["data"].keys()
                if key.startswith("demo_")
            ]

        # Store episodes with observations and actions
        self.episodes = []
        for demo_key in self.demo_keys:
            demo = self.hdf5_file[f"data/{demo_key}"]
            if self.use_states:
                obs = demo["states"][:]
            else:
                obs = self._get_obs(demo)
            actions = demo["actions"][:]
            dones = demo["dones"][:] if "dones" in demo else np.zeros(len(actions))

            if len(obs) > 1:  # Need at least 2 states
                self.episodes.append({
                    'obs': obs.astype(np.float32),
                    'actions': actions.astype(np.float32),
                    'dones': dones,
                })

    def _get_obs(self, demo):
        """Concatenate observation keys into a single observation vector."""
        obs_group = demo["obs"]
        keys = self.obs_keys if self.obs_keys else ROBOMIMIC_DEFAULT_OBS_KEYS

        obs_arrays = []
        for key in keys:
            if key in obs_group:
                arr = obs_group[key][:]
                if arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0], -1)
                obs_arrays.append(arr)

        if not obs_arrays:
            return demo["states"][:]

        return np.concatenate(obs_arrays, axis=-1)

    def get_trajectory(self):
        """Return a random trajectory for visualization."""
        episode_ind = np.random.randint(0, len(self.episodes))
        ep = self.episodes[episode_ind]
        obs = ep['obs']
        traj = torch.from_numpy(obs)
        g = traj[-1]  # (D,)
        g_rep = g.unsqueeze(0).expand(traj.shape[0], -1)  # (T+1, D)
        return torch.cat([traj, g_rep], dim=-1)

    def _sample(self):
        episode_ind = np.random.randint(0, len(self.episodes))
        ep = self.episodes[episode_ind]
        obs = ep['obs']  # (T+1, obs_dim)
        actions = ep['actions']  # (T, action_dim)
        dones = ep['dones']
        episode_len = len(obs) - 1  # T transitions

        if episode_len < 2:
            return self._sample()

        # Sample (o_t, a_t, o_t+1, o_goal) for IQL training
        t = np.random.randint(0, episode_len - 1)
        t_g = np.random.randint(t + 1, episode_len)

        g = torch.from_numpy(obs[t_g])
        s = torch.from_numpy(obs[t])
        s_next = torch.from_numpy(obs[t + 1])
        a = torch.from_numpy(actions[t])

        reached = t + 1 == t_g
        is_terminal = bool(dones[t]) if t < len(dones) else False
        discount = torch.tensor(0.0 if is_terminal or reached else 1.0, dtype=torch.float32)
        r = torch.tensor(-1.0, dtype=torch.float32)

        # Concatenate state and goal
        ob = torch.cat([s, g], dim=-1)
        ob_next = torch.cat([s_next, g], dim=-1)
        return (ob, a, r, discount, ob_next)

    def __iter__(self):
        while True:
            yield self._sample()

    def __del__(self):
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()
