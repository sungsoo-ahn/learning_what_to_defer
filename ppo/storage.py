import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import dgl
import numpy as np
import random

class RolloutStorage(object):
    def __init__(
        self, 
        max_t, 
        batch_size, 
        num_samples
        ):
        self.max_t = max_t
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.step_t = 0
        
        self.obs = []
        self.actions = []
        self.action_log_probs = []   
        
        self.value_preds = torch.zeros(
            max_t + 1, 
            batch_size, 
            num_samples 
            )
        self.rewards = torch.zeros(
            max_t, 
            batch_size, 
            num_samples
            )
        self.masks = torch.ones(
            max_t + 1, 
            batch_size, 
            num_samples
            )
        self.rets = torch.zeros(
            max_t + 1,
            batch_size, 
            num_samples
            )
        self.advantages = torch.zeros(
            max_t,
            batch_size,
            num_samples
            )        
        
    def insert_ob_and_g(self, ob, g):
        ob_shape = list(ob.size())
        num_nodes = ob_shape[0]
        num_samples = ob_shape[1]
        self.g = g
        self.obs = torch.zeros(
            self.max_t + 1, 
            *ob_shape
            )
        self.actions = torch.zeros(
            self.max_t, 
            num_nodes,
            num_samples,
            dtype = torch.long
            )
        self.action_log_probs = torch.zeros(
            self.max_t, 
            num_nodes,
            num_samples
            )
        self.obs[0].copy_(ob)
        self.masks.fill_(0)
        self.masks[0].fill_(1)
        self.step_t = 0
    
    def insert_tensors(
        self, 
        ob, 
        action, 
        action_log_prob, 
        value_pred, 
        reward, 
        done
        ):
        ob_ = ob.cpu()
        action_ = action.cpu()
        action_log_prob_ = action_log_prob.cpu()
        value_pred_ = value_pred.cpu()
        reward_ = reward.cpu()
        done_ = done.cpu()

        if self.step_t == self.max_t:
            self.step_t = 0

        self.obs[self.step_t + 1].copy_(ob_)
        self.actions[self.step_t].copy_(action_)
        self.action_log_probs[self.step_t].copy_(action_log_prob_)
        self.value_preds[self.step_t].copy_(value_pred_)
        self.rewards[self.step_t].copy_(reward_)
        self.masks[self.step_t + 1].copy_(~done_)
        self.step_t = self.step_t + 1
        
    def compute_rets_and_advantages(self, gamma):
        for t in reversed(range(self.max_t)):
            self.rets[t] = (
                self.rewards[t] 
                + gamma * self.masks[t + 1] * self.rets[t + 1]
                )
        advantages = self.rets[:-1] - self.value_preds[:-1]
        self.advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5
            )

    def build_update_sampler(self, optim_batch_size, optim_num_samples):
        num_nodes = self.g.number_of_nodes()
        batch_size = self.g.batch_size
        flat_obs = (
            self.obs
            .narrow(0, 0, self.step_t)
            .permute(0, 2, 1, 3)
            .reshape(-1, num_nodes, self.obs.size(3))
            )
        flat_actions = (
            self.actions
            .narrow(0, 0, self.step_t)
            .permute(0, 2, 1)
            .reshape(-1, num_nodes)
            )
        flat_action_log_probs = (
            self.action_log_probs
            .narrow(0, 0, self.step_t)
            .permute(0, 2, 1)
            .reshape(-1, num_nodes)
            )
        flat_value_preds = (
            self.value_preds
            .narrow(0, 0, self.step_t)
            .permute(0, 2, 1)
            .reshape(-1, batch_size)
            )
        flat_rets = (
            self.rets
            .narrow(0, 0, self.step_t)
            .permute(0, 2, 1)
            .reshape(-1, batch_size)
            )
        flat_advantages = (
            self.advantages
            .narrow(0, 0, self.step_t)
            .permute(0, 2, 1)
            .reshape(-1, batch_size)
            )
        flat_dim = flat_obs.size(0)

        sampler = BatchSampler(
            SubsetRandomSampler(range(flat_dim)),
            min(flat_dim, optim_batch_size), 
            drop_last = False
            )
        
        sampler_t = 0
        while sampler_t < optim_num_samples:
            for idx in sampler:
                yield (
                    self.g,
                    flat_obs[idx], 
                    flat_actions[idx], 
                    flat_action_log_probs[idx], 
                    flat_value_preds[idx], 
                    flat_rets[idx], 
                    flat_advantages[idx]
                    )
                
                sampler_t += 1
                if sampler_t == optim_num_samples:
                    break