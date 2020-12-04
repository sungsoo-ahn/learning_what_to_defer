import math
import torch
from torch.optim import Adam
import dgl

class ProxPolicyOptimFramework(object):
    def __init__(
        self, 
        actor_critic,
        init_lr,  
        clip_value, 
        optim_num_samples,
        optim_batch_size, 
        critic_loss_coef, 
        reg_coef, 
        max_grad_norm, 
        device
        ):

        self.actor_critic = actor_critic
        self.optimizer = Adam(actor_critic.parameters(), lr=init_lr)
        self.clip_value = clip_value
        self.optim_num_samples = optim_num_samples
        self.optim_batch_size = optim_batch_size
        self.critic_loss_coef = critic_loss_coef
        self.reg_coef = reg_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
    
    def update(self, rollout):
        avg_actor_loss = torch.tensor(0.0)
        avg_critic_loss = torch.tensor(0.0)
        avg_entropy = torch.tensor(0.0)
        cnt = 0
        data_loader = rollout.build_update_sampler(
            self.optim_batch_size,
            self.optim_num_samples
            )
        for samples in data_loader:
            (g,
            obs, 
            actions, 
            old_action_log_probs, 
            old_value_preds, 
            rets, 
            old_advantages
            ) = samples
            
            obs = obs.to(self.device)
            node_masks = (obs.select(2, 0).long() == 2)

            actions = actions.to(self.device)
            old_action_log_probs = old_action_log_probs.to(self.device)
            old_value_preds = old_value_preds.to(self.device)
            rets = rets.to(self.device)
            old_advantages = old_advantages.to(self.device)

            (action_log_probs, 
            entropy,
            value_preds,
            node_masks
            ) = self.actor_critic.evaluate_batch(
                obs.permute(1, 0, 2), 
                g, 
                actions.permute(1, 0)
                )
            
            action_log_probs = action_log_probs.permute(1, 0)
            value_preds = value_preds.permute(1, 0)
            node_masks = node_masks.permute(1, 0)
            
            # compute actor loss
            diff = action_log_probs - old_action_log_probs
            clamped_diff = torch.clamp(
                diff,
                math.log(1.0 - self.clip_value), 
                math.log(1.0 + self.clip_value)
                )
            stacked_diff = torch.stack(
                [diff, clamped_diff], dim = 2
                )
            g = g.to(self.device)
            g.ndata['h'] = stacked_diff.permute(1, 0, 2) 
            h = dgl.sum_nodes(g, 'h').permute(1, 0, 2)
            g.ndata.pop('h')
            ratio = torch.exp(h.select(2, 0))
            clamped_ratio = torch.exp(h.select(2, 1))
            surr1 = ratio * old_advantages
            surr2 = clamped_ratio * old_advantages
            actor_loss = - torch.min(surr1, surr2).mean()
            
            # compute entropy loss
            reg_loss = - entropy
        
            # compute critic loss
            critic_loss = 0.5 * (value_preds - rets).pow(2).mean()
        
            # add up losses and back prop
            loss = (
                actor_loss 
                + self.critic_loss_coef * critic_loss 
                + self.reg_coef * reg_loss
                )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), 
                self.max_grad_norm
                )
            self.optimizer.step()
            
            # avg_actor_loss += actor_loss.detach()
            # avg_critic_loss += critic_loss.detach()
            # avg_entropy += entropy.detach()
            avg_actor_loss += actor_loss.to("cpu")
            avg_critic_loss += critic_loss.to("cpu")
            avg_entropy += entropy.to("cpu")
            cnt += 1
            torch.cuda.empty_cache()

        avg_actor_loss /= cnt 
        avg_critic_loss /= cnt 
        avg_entropy /= cnt
        
        return avg_actor_loss, avg_critic_loss, avg_entropy