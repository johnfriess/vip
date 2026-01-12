# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from pathlib import Path
from torchvision.utils import save_image
import time
import copy
import torchvision.transforms as T
from vip.utils.utils import STATE_DATASETS

epsilon = 1e-8
def do_nothing(x): return x

class Trainer():
    def __init__(self, eval_freq):
        self.eval_freq = eval_freq

    def update(self, model, batch, step, eval=False, datasource='ego4d'):
        t0 = time.time()
        metrics = dict()
        if eval:
            model.eval()
        else:
            model.train()

        t1 = time.time()
        ## Batch
        b_f, b_reward = batch
        t2 = time.time()

        ## Encode Start and End Frames
        bs = b_f.shape[0]
        stack_size = b_f.shape[1]
        if datasource not in STATE_DATASETS:
            H = b_f.shape[-2]
            W = b_f.shape[-1]
            b_im_r = b_f.reshape(bs*stack_size, 3, H, W)
            alles = model(b_im_r)
        else:
            state_dim = b_f.shape[-1]
            b_st = b_f.reshape(bs*stack_size, state_dim)
            alles = model(b_st)

        alle = alles.reshape(bs, stack_size, -1)
        e0 = alle[:, 0] # initial, o_0
        eg = alle[:, 1] # final, o_g
        es0_vip = alle[:, 2] # o_t
        es1_vip = alle[:, 3] # o_t+1

        full_loss = 0

        ## LP Loss
        l2loss = torch.linalg.norm(alles, ord=2, dim=-1).mean()
        l1loss = torch.linalg.norm(alles, ord=1, dim=-1).mean()
        metrics['l2loss'] = l2loss.item()
        metrics['l1loss'] = l1loss.item()
        full_loss += model.module.l2weight * l2loss
        full_loss += model.module.l1weight * l1loss
        t3 = time.time()

        ## VIP Loss 
        V_0 = model.module.sim(e0, eg) # -||phi(s) - phi(g)||_2
        r =  b_reward.to(V_0.device) # R(s;g) = (s==g) - 1 
        V_s = model.module.sim(es0_vip, eg)
        V_s_next = model.module.sim(es1_vip, eg)
        V_loss = (1-model.module.gamma) * -V_0.mean() + torch.log(epsilon + torch.mean(torch.exp(-(r + model.module.gamma * V_s_next - V_s))))

        # Optionally, add additional "negative" observations
        V_s_neg = []
        V_s_next_neg = []
        for _ in range(model.module.num_negatives):
            perm = torch.randperm(es0_vip.size()[0])
            es0_vip_shuf = es0_vip[perm]
            es1_vip_shuf = es1_vip[perm]

            V_s_neg.append(model.module.sim(es0_vip_shuf, eg))
            V_s_next_neg.append(model.module.sim(es1_vip_shuf, eg))

        if model.module.num_negatives > 0:
            V_s_neg = torch.cat(V_s_neg)
            V_s_next_neg = torch.cat(V_s_next_neg)
            r_neg = -torch.ones(V_s_neg.shape).to(V_0.device)
            V_loss = V_loss + torch.log(epsilon + torch.mean(torch.exp(-(r_neg + model.module.gamma * V_s_next_neg - V_s_neg))))
        
        metrics['vip_loss'] = V_loss.item()
        full_loss += V_loss
        metrics['full_loss'] = full_loss.item()
        t4 = time.time()

        if not eval:
            model.module.encoder_opt.zero_grad()
            full_loss.backward()
            model.module.encoder_opt.step()
        t5 = time.time()

        st = f"Load time {t1-t0}, Batch time {t2-t1}, Encode and LP time {t3-t2}, VIP time {t4-t3}, Backprop time {t5-t4}"
        return metrics,st

class IQLTrainer():
    def __init__(self, eval_freq):
        self.eval_freq = eval_freq

    def update(self, model, batch, step, eval=False, datasource='ego4d'):
        t0 = time.time()
        metrics = dict()
        if eval:
            model.eval()
        else:
            model.train()

        t1 = time.time()
        ## Batch
        b_s, b_a, b_r, b_discount, b_s_next = batch
        t2 = time.time()

        ## Update V network
        with torch.no_grad():
            q1, q2 = model.module.q_values(b_s, b_a)
            q_min = torch.min(q1, q2)
        v = model.module.v_value(b_s)
        diff = q_min - v
        weight = torch.where(diff < 0, 1 - model.module.expectile, model.module.expectile)
        v_loss = torch.mean(weight * diff.pow(2))
        if not eval:
            model.module.v_optimizer.zero_grad()
            v_loss.backward()
            model.module.v_optimizer.step()
        metrics["v_loss"] = v_loss.item()
        t3 = time.time()

        ## Update Q networks
        with torch.no_grad():
            target_q = b_r + b_discount * model.module.gamma * model.module.v_value(b_s_next)
        q1, q2 = model.module.q_values(b_s, b_a)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss
        if not eval:
            model.module.q_optimizer.zero_grad()
            q_loss.backward()
            model.module.q_optimizer.step()
        metrics["q_loss"] = q_loss.item()
        t4 = time.time()

        ## Update policy
        with torch.no_grad():
            q1, q2 = model.module.q_values(b_s, b_a)
            q_min = torch.min(q1, q2)
            adv = q_min - model.module.v_value(b_s)
            weights = torch.clamp(torch.exp(model.module.beta * adv), max=model.module.max_adv)
        dist = model.module.policy_dist(b_s)
        log_prob = dist.log_prob(b_a).sum(dim=-1)
        pi_loss = torch.mean(-(weights * log_prob))
        if not eval:
            model.module.pi_optimizer.zero_grad()
            pi_loss.backward()
            model.module.pi_optimizer.step()
        metrics["pi_loss"] = pi_loss.item()
        t5 = time.time()

        st = f"Load time {t1-t0}, Batch time {t2-t1}, Q time {t3-t2}, V time {t4-t3}, Policy time {t5-t4}"
        return metrics,st
