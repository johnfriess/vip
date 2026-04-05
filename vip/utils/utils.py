# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from vip.utils.data_loaders import (
    STATE_DATASETS, KITCHEN_IMAGE_ALL, VIPBuffer, StateVIPBuffer, StateIQLBuffer,
    RobomimicVIPBuffer, RobomimicIQLBuffer
)
from vip.models.model_derail import StateMultilinearDERAIL, StateEuclideanDERAIL, ImageEuclideanDERAIL
from vip.trainer import VIPTrainer, IQLTrainer, DERAILTrainer

class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

def create_buffer(model_type='vip', datasource='ego4d', datapath=None, num_workers=10, doaug="none",
                  use_achieved_goal=False, obs_keys=None, filter_key=None, use_states=False,
                  frame_stack=1, frame_combine=False):
    if model_type in ['vip', 'derail']:
        if datasource in ('kitchen-image', KITCHEN_IMAGE_ALL):
            return VIPBuffer(datasource, datapath, num_workers, doaug, frame_stack, frame_combine)
        elif datapath and datapath.endswith('.hdf5'):
            return RobomimicVIPBuffer(
                hdf5_path=datapath,
                obs_keys=obs_keys,
                filter_key=filter_key,
                use_states=use_states
            )
        elif datasource in STATE_DATASETS:
            return StateVIPBuffer(datasource, use_achieved_goal=use_achieved_goal)
        else:
            return VIPBuffer(datasource, datapath, num_workers, doaug, frame_stack)
    elif model_type == 'iql':
        if datapath and datapath.endswith('.hdf5'):
            return RobomimicIQLBuffer(
                hdf5_path=datapath,
                obs_keys=obs_keys,
                filter_key=filter_key,
                use_states=use_states
            )
        else:
            return StateIQLBuffer(datasource)

def create_trainer(model_type='vip', eval_freq=1000):
    if model_type == 'vip':
        return VIPTrainer(eval_freq)
    elif model_type == 'iql':
        return IQLTrainer(eval_freq)
    elif model_type == 'derail':
        return DERAILTrainer(eval_freq)

def visualize_trajectory(model, model_type, datasource, buffer, device, datapath=None, output_path=None):
    is_robomimic = datapath and datapath.endswith('.hdf5')
    if not is_robomimic and datasource not in STATE_DATASETS and datasource not in ('kitchen-image', KITCHEN_IMAGE_ALL):
        return

    traj = buffer.get_trajectory().to(device)
    with torch.no_grad():
        if model_type == 'vip':
            etraj = model(traj)
            eg = etraj[-1]
            values = model.module.sim(etraj, eg).cpu()
        elif model_type == 'iql':
            values = model.module.v_value(traj).cpu()
        elif model_type == "derail":
            if isinstance(model.module, StateMultilinearDERAIL):
                g = traj[-1]
                g_rep = g.unsqueeze(0).expand(traj.shape[0], -1)
                h1, h2, h3 = model(traj, g_rep)
                values = model.module.value(h1, h2, h3).cpu()
            elif isinstance(model.module, (StateEuclideanDERAIL, ImageEuclideanDERAIL)):
                etraj = model(traj)
                eg = etraj[-1]
                values = model.module.sim(etraj, eg).cpu()

    plt.figure()
    plt.plot(values)
    plt.xlabel("Frame")
    plt.ylabel(f"{model_type.upper()} Value")
    plt.title(f"{model_type.upper()} Value along Expert Trajectory ({datasource})")
    plt.savefig(output_path or f"{model_type}_trajectory_visualization.png")
    plt.close()


def visualize_trajectory_with_frames(model, model_type, datasource, buffer, device,
                                     goal_fraction=0.25, num_sampled_frames=10,
                                     datapath=None, output_path=None):
    is_robomimic = datapath and datapath.endswith('.hdf5')
    if not is_robomimic and datasource not in STATE_DATASETS and datasource not in ('kitchen-image', KITCHEN_IMAGE_ALL):
        return

    traj = buffer.get_trajectory().to(device)
    traj_len = traj.shape[0]
    goal_idx = min(int(goal_fraction * (traj_len - 1)), traj_len - 1)

    with torch.no_grad():
        etraj = model(traj)
        eg = etraj[goal_idx]
        if model_type == 'vip':
            values = model.module.sim(etraj, eg).cpu()
        elif model_type == 'derail' and isinstance(model.module, (StateEuclideanDERAIL, ImageEuclideanDERAIL)):
            values = model.module.sim(etraj, eg).cpu()
        else:
            return

    values_np = values[:goal_idx + 1].numpy()

    # Pick evenly-spaced frame indices up to the goal
    frame_indices = np.linspace(0, goal_idx, num_sampled_frames, dtype=int)

    is_state = datasource in STATE_DATASETS or (is_robomimic)
    raw_frames = traj.cpu()

    if is_state:
        # State-based: value curve only (no images to display)
        fig, ax_val = plt.subplots(figsize=(10, 4))
    else:
        # Image-based: frames on top row, value curve below
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(2, num_sampled_frames, height_ratios=[1, 2], hspace=0.3)
        for i, fidx in enumerate(frame_indices):
            ax_img = fig.add_subplot(gs[0, i])
            frame = raw_frames[fidx].permute(1, 2, 0).numpy()
            if frame.max() > 2.0:
                frame = frame / 255.0
            ax_img.imshow(np.clip(frame[..., -3:], 0, 1))
            ax_img.set_title(f"f{fidx}", fontsize=8)
            ax_img.axis('off')
        ax_val = fig.add_subplot(gs[1, :])

    ax_val.plot(range(goal_idx + 1), values_np, linewidth=1.5)
    for fidx in frame_indices:
        ax_val.axvline(x=fidx, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
        ax_val.plot(fidx, values_np[fidx], 'o', color='red', markersize=5)

    ax_val.axvline(x=goal_idx, color='r', linestyle='--', alpha=0.7, label=f'goal @ {goal_idx}')
    ax_val.set_xlabel("Frame")
    ax_val.set_ylabel(f"{model_type.upper()} Value")
    ax_val.set_title(f"{model_type.upper()} Value → Goal @ {int(goal_fraction*100)}% (frame {goal_idx})")
    ax_val.legend()

    plt.savefig(output_path or f"{model_type}_frames_value.png", dpi=150, bbox_inches='tight')
    plt.close()
