# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Identity
import torchvision
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image
import torchvision.transforms as T

class StateIQL(nn.Module):
    def __init__(self, device="cuda", lr=1e-4, state_dim=60, action_dim=9, hidden_dim=256, size=1, mlp_width=256, gamma=0.98, expectile=0.7, beta=3.0, max_adv=100.0):
        super().__init__()
        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.size = size
        self.gamma = gamma
        self.expectile = expectile
        self.beta = beta
        self.max_adv = max_adv

        layers = [nn.LayerNorm(state_dim), nn.Linear(state_dim, mlp_width), nn.ReLU()]
        for _ in range(size):
            layers.append(nn.Linear(mlp_width, mlp_width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(mlp_width, hidden_dim))

        self.encoder = nn.Sequential(*layers)

        # Q networks: Q1(s, a), Q2(s, a)
        def make_q():
            return nn.Sequential(
                nn.Linear(hidden_dim + action_dim, mlp_width),
                nn.ReLU(),
                nn.Linear(mlp_width, mlp_width),
                nn.ReLU(),
                nn.Linear(mlp_width, 1),
            )
        self.q1 = make_q()
        self.q2 = make_q()

        # V network: V(s)
        self.v = nn.Sequential(
            nn.Linear(hidden_dim, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, 1),
        )

        # Policy network: π(a | s)
        self.policy_mean = nn.Sequential(
            nn.Linear(hidden_dim, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        ## Optimizers
        self.q_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=lr,
        )
        self.v_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.v.parameters()),
            lr=lr)
        self.pi_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.policy_mean.parameters()) + [self.log_std],
            lr=lr,
        )

    ## Forward Call (state --> representation)
    def forward(self, obs):
        h = self.encoder(obs)
        return h

    def q_values(self, s, a):
        z = self.forward(s)
        sa = torch.cat([z, a], dim=-1)
        q1 = self.q1(sa).squeeze(-1)
        q2 = self.q2(sa).squeeze(-1)
        return (q1, q2)

    def v_value(self, s):
        z = self.forward(s)
        return self.v(z).squeeze(-1)

    def policy_dist(self, s):
        z = self.forward(s)
        mean = self.policy_mean(z)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)
