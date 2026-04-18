import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

from derail.utils import DEFAULT_DEVICE, compute_batched, update_exponential_moving_average


EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class DVL_ODICE(nn.Module):
    def __init__(self,  qf,vf, optimizer_factory,
                 tau, beta,lamda, eta, lr=3e-4, discount=0.99, alpha=0.005, reconstruction_alpha=0.01):
        super().__init__()
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.v_target = copy.deepcopy(vf).requires_grad_(False).to(DEFAULT_DEVICE)
        # self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = optimizer_factory(self.vf.parameters(), lr = lr)
        self.q_optimizer = optimizer_factory(self.qf.parameters(), lr = lr)
        # self.policy_optimizer = optimizer_factory(self.policy.parameters())
        # self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha
        self.lamda = lamda
        self.eta=eta
        self.reconstruction_alpha = reconstruction_alpha

    def f_star(self, residual):
        omega_star = torch.max(residual / 2 + 1, torch.zeros_like(residual))
        return residual * omega_star - (omega_star - 1)**2

    def update(self, observations, next_observations, goals, gt_s0, gt_s1, rewards, terminals):
        # print("Updating with SMORE on steroids")
        v_loss_val = 0.0
        q_loss_val = 0.0
        # with torch.no_grad():
        beta = 0.5
        lamda = self.lamda
        ita = self.eta
        v_curr =  self.vf(observations, goals)
        v_next =  self.vf(next_observations, goals)
        v_curr_target = self.v_target(observations, goals)
        v_next_target = self.v_target(next_observations, goals)

        # gt_v_curr = self.vf(gt_s0, gt_s1)
        # gt_v_next = self.vf(gt_s1, gt_s1)
        # gt_v_curr_target = self.v_target(gt_s0, gt_s1)
        # gt_v_next_target = self.v_target(gt_s1, gt_s1)
        # import ipdb; ipdb.set_trace()
        # Update value function
        backward_residual = rewards+(1. - terminals.float()) * self.discount * v_next - v_curr_target
        forward_residual =  rewards+(1. - terminals.float()) * self.discount * v_next_target - v_curr
        # print(rewards.sum(),rewards.shape[0])
        # import ipdb;ipdb.set_trace()
        # gt_backward_residual = rewards+(1. - terminals.float()) * self.discount * gt_v_next - gt_v_curr_target
        # gt_forward_residual =  rewards+(1. - terminals.float()) * self.discount * gt_v_next_target - gt_v_curr
        
        backward_dual_loss = ita*lamda*self.f_star(backward_residual).mean()
        forward_dual_loss = lamda*self.f_star(forward_residual).mean()

        # backward_dual_loss = (beta * ita* self.f_star(gt_backward_residual) + (1-beta) * self.f_star(backward_residual) - (1-beta)*backward_residual).mean()
        # forward_dual_loss = (beta * self.f_star(gt_forward_residual) + (1-beta) * self.f_star(forward_residual) - (1-beta)*forward_residual).mean()
        v_loss_val += 0.5*(forward_dual_loss.item() + backward_dual_loss.item())
        self.v_optimizer.zero_grad(set_to_none=True)
        forward_grad_list, backward_grad_list = [], []
        forward_dual_loss.backward(retain_graph=True)
        for param in list(self.vf.parameters()):
            if param.grad is None:
                # import ipdb; ipdb.set_trace()
                continue
            forward_grad_list.append(param.grad.clone().detach().reshape(-1))
        backward_dual_loss.backward()
        forward_id = 0
        for i, param in enumerate(list(self.vf.parameters())):
            if param.grad is None:
                # import ipdb; ipdb.set_trace()
                continue
            backward_grad_list.append(param.grad.clone().detach().reshape(-1) - forward_grad_list[forward_id])
            forward_id+=1
        forward_grad, backward_grad = torch.cat(forward_grad_list), torch.cat(backward_grad_list)
        parallel_coef = (torch.dot(forward_grad, backward_grad) / max(torch.dot(forward_grad, forward_grad),
                                                                      1e-10)).item()  # avoid zero grad caused by f*
        forward_grad = (1 - parallel_coef) * forward_grad + backward_grad

        param_idx = 0
        for i, grad in enumerate(forward_grad_list):
            forward_grad_list[i] = forward_grad[param_idx: param_idx + grad.shape[0]]
            param_idx += grad.shape[0]
        # reset gradient and calculate
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss =  (1-lamda)*v_curr.mean()
        v_loss.backward()
        forward_id = 0
        for i, param in enumerate(list(self.vf.parameters())):
            if param.grad is None:
                # import ipdb; ipdb.set_trace()
                continue
            param.grad += forward_grad_list[forward_id].reshape(param.grad.shape)
            forward_id+=1

        self.v_optimizer.step()

        v_loss_val += v_loss.item()
        # # Update target Q network
        update_exponential_moving_average(self.v_target, self.vf, self.alpha)

        return v_loss_val, q_loss_val