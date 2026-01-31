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

STATE_DATASETS = [
    "D4RL/kitchen/complete-v2",
    "D4RL/kitchen/partial-v2",
    "D4RL/kitchen/mixed-v2",
]

def get_ind(vid, index, ds="ego4d"):
    if ds == "ego4d":
        return torchvision.io.read_image(f"{vid}{index:06}.jpg")
    else:
        try:
            return torchvision.io.read_image(f"{vid}/{index}.jpg")
        except: 
            return torchvision.io.read_image(f"{vid}/{index}.png")

## Data Loader for VIP
class VIPBuffer(IterableDataset):
    def __init__(self, datasource='ego4d', datapath=None, num_workers=10, doaug = "none"):
        self._num_workers = max(1, num_workers)
        self.datasource = datasource
        self.datapath = datapath
        assert(datapath is not None)
        self.doaug = doaug
        
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

    def _sample(self):
        # Sample a video from datasource
        if self.datasource == 'ego4d':
            vidid = np.random.randint(0, self.ego4dlen)
            m = self.manifest.iloc[vidid]
            vidlen = m["len"]
            vid = m["path"]
        else: 
            video_paths = glob.glob(f"{self.datapath}/[0-9]*")
            num_vid = len(video_paths)

            video_id = np.random.randint(0, int(num_vid)) 
            vid = f"{video_paths[video_id]}"

            # Video frames must be .png or .jpg
            vidlen = len(glob.glob(f'{vid}/*.png'))
            if vidlen == 0:
                vidlen = len(glob.glob(f'{vid}/*.jpg'))

        # Sample (o_t, o_k, o_k+1, o_T) for VIP training
        start_ind = np.random.randint(0, vidlen-2)  
        end_ind = np.random.randint(start_ind+1, vidlen)

        s0_ind_vip = np.random.randint(start_ind, end_ind)
        s1_ind_vip = min(s0_ind_vip+1, end_ind)
        
        # Self-supervised reward (this is always -1)
        reward = float(s0_ind_vip == end_ind) - 1

        if self.doaug == "rctraj":
            ### Encode each image in the video at once the same way
            im0 = get_ind(vid, start_ind, self.datasource) 
            img = get_ind(vid, end_ind, self.datasource)
            imts0_vip = get_ind(vid, s0_ind_vip, self.datasource)
            imts1_vip = get_ind(vid, s1_ind_vip, self.datasource)
            
            allims = torch.stack([im0, img, imts0_vip, imts1_vip], 0)
            allims_aug = self.aug(allims / 255.0) * 255.0

            im0 = allims_aug[0]
            img = allims_aug[1]
            imts0_vip = allims_aug[2]
            imts1_vip = allims_aug[3]

        else:
            ### Encode each image individually
            im0 = self.aug(get_ind(vid, start_ind, self.datasource) / 255.0) * 255.0
            img = self.aug(get_ind(vid, end_ind, self.datasource) / 255.0) * 255.0
            imts0_vip = self.aug(get_ind(vid, s0_ind_vip, self.datasource) / 255.0) * 255.0
            imts1_vip = self.aug(get_ind(vid, s1_ind_vip, self.datasource) / 255.0) * 255.0

        im = torch.stack([im0, img, imts0_vip, imts1_vip])
        im = self.preprocess(im)
        return (im, reward)

    def __iter__(self):
        while True:
            yield self._sample()

class StateVIPBuffer(IterableDataset):
    def __init__(self, datasource="D4RL/kitchen/complete-v2"):
        import minari

        dataset = minari.load_dataset(datasource, download=True)

        # Store episode observations directly (list of numpy arrays)
        # Each episode.observations['observation'] has shape (T+1, obs_dim)
        # Minari kitchen obs is 59D (goals stored separately, not embedded)
        self.episodes = [
            ep.observations['observation']
            for ep in dataset.iterate_episodes()
            if len(ep.observations['observation']) > 2  # Need at least 3 states
        ]

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

        dataset = minari.load_dataset(datasource, download=True)

        # Store EpisodeData objects directly
        # Minari kitchen obs is 59D (goals stored separately, not embedded)
        self.episodes = [
            ep for ep in dataset.iterate_episodes()
            if len(ep.observations['observation']) > 1  # Need at least 2 states
        ]

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
