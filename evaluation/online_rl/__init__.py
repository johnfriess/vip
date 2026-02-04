# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Online RL evaluation for VIP models.

This module provides:
- VIPRewardEnv: Environment wrapper that computes VIP embedding rewards
- run_online_rl: Training script for MJRL (NPG/PPO) with VIP rewards
"""

from .vip_reward_env import VIPRewardEnv, make_vip_env

__all__ = ['VIPRewardEnv', 'make_vip_env']