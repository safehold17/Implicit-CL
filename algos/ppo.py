# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random 

class PPO():
    """
    Vanilla PPO
    """
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 kl_loss_coef=0.0,
                 use_ego_ctrlsim_kl_loss=False,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 clip_value_loss=True,
                 log_grad_norm=False):

        self.actor_critic = actor_critic    

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.clip_value_loss = clip_value_loss

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.kl_loss_coef = kl_loss_coef

        # new ego_ctrlsim policy vs student policy KL
        self.use_ego_ctrlsim_kl_loss = use_ego_ctrlsim_kl_loss

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        # `.parameters()` is a method provided by `nn.Module` in PyTorch that returns an iterator over all trainable parameters of a model.

        self.log_grad_norm = log_grad_norm

    def _grad_norm(self):
        total_norm = 0
        for p in self.actor_critic.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)  #Compute the L2 norm of the gradient of a single parameter.
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def update(self, rollouts, discard_grad=False, kl_dict=None):
        # online model-vs-model KL path, not using in this project
        use_model_kl_loss = (
            (kl_dict is not None)
            and (self.kl_loss_coef > 0.0)
            and (discard_grad is False)
        )
        
        # When `use_popart=True`, the model outputs are in a normalized space; 
        # therefore, `denorm_value_preds` must be applied to map them back to the true value scale (e.g., reward or return-to-go). 
        # Otherwise, severe scaling errors will occur.
        if rollouts.use_popart:
            value_preds = rollouts.denorm_value_preds
        else:
            value_preds = rollouts.value_preds

        advantages = rollouts.returns[:-1] - value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)  # standard normalization of advantages + small constant for numerical stability

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        if use_model_kl_loss:
            kl_loss_epoch = 0
    
        # Running accumulator for the ego_ctrlsim teacher-vs-student KL term.
        ego_ctrlsim_kl_loss_epoch = 0

        if self.log_grad_norm:
            grad_norms = []

        for e in range(self.ppo_epoch):
        # The `feed_forward_generator` flattens the rollout into independent samples and randomly forms mini-batches for training a feedforward policy.
        # The `recurrent_generator` samples complete time sequences on a per-environment basis and provides the corresponding initial hidden states for training RNN/LSTM-based policies.
            if self.actor_critic.is_recurrent:  # not using recurrent policy in this project
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                if self.actor_critic.is_recurrent:
                    # not using a recurrent policy
                    raise NotImplementedError("recurrent PPO update currently unsupported")
                else:
                    obs_batch, recurrent_hidden_states_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ, ego_ctrlsim_action_logits_batch, ego_ctrlsim_valid_batch = sample

                has_valid_ego_ctrlsim_teacher = (
                    ego_ctrlsim_action_logits_batch is not None
                    and ego_ctrlsim_valid_batch is not None
                    and bool(ego_ctrlsim_valid_batch.any().item())
                )

                # ego_ctrlsim teacher logits come from rollout storage; enable this path only when the sample carries them.
                # Teacher logits are fixed from rollout storage; the student distribution
                # must be recomputed on every PPO mini-batch update using current parameters.
                use_ego_ctrlsim_kl_loss = (
                    self.use_ego_ctrlsim_kl_loss
                    and
                    has_valid_ego_ctrlsim_teacher
                    and (self.kl_loss_coef > 0.0)
                    and (discard_grad is False)
                )

                # 一旦需要任一 KL 项，就必须返回当前 student 的完整分布对象
                # Once any KL term is active, request the current full student distribution.
                if use_model_kl_loss or use_ego_ctrlsim_kl_loss:
                    values, action_log_probs, dist_entropy, _, dist_protagonist = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch, return_policy_logits=True)
                    if use_model_kl_loss:
                        # 旧的在线 KL 使用对端模型当前分布，但不回传对端梯度
                        # The old online KL uses model's current distribution without backpropagating through it.
                        with torch.no_grad():
                            _, _, _, _, dist_antagonist = kl_dict['antagonist_model'].evaluate_actions(
                            obs_batch, recurrent_hidden_states_batch, masks_batch,
                            actions_batch, return_policy_logits=True)
                else:
                    values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch)

                # ratio for importance sampling, also the basis for PPO clipping and surrogate loss calculation    
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch) 
                # action_log_probs comes from the current policy, student model
                # old_action_log_probs_batch comes from the rollout collection policy, no mini-batch, [num_steps, num_processes, 1]

                surr1 = ratio * adv_targ  #advantage value for the current mini-batch samples, scaled by the importance sampling ratio
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if rollouts.use_popart:  # default is False
                    self.actor_critic.popart.update(return_batch)
                    return_batch = self.actor_critic.popart.normalize(return_batch)

                if self.clip_value_loss:  # default is False
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                    value_losses_clipped).mean()
                else:
                    value_loss = F.smooth_l1_loss(values, return_batch)
                    
                if use_model_kl_loss:
                    # keep the existing online model-vs-model KL behavior.
                    kl_div = torch.distributions.kl.kl_divergence(dist_antagonist, dist_protagonist)
                    # protagonist 是 student，antagonist 是固定的 adversary
                    # 计算 KL(teacher || student) 以匹配旧指标，但只反向传播 student
                    bs = kl_div.shape[0]
                    kl_loss = kl_div.sum() / bs
                
                if use_ego_ctrlsim_kl_loss:
                    # 使用 rollout 时缓存的 teacher logits 构造固定 teacher 分布
                    # Build the fixed teacher distribution from teacher logits cached during rollout.
                    # Compute KL(teacher || student) with fixed rollout teacher logits
                    # and the current student distribution from this update step.
                    ego_ctrlsim_valid_mask = ego_ctrlsim_valid_batch.squeeze(-1).bool()
                    # 仅对 rollout 中真正做过 ego teacher 推理的 timestep 计算 KL。
                    # 稀疏 KL 下，无效 timestep 只是“没有 teacher 样本”，不能把它们当成全零 logits；否则会把缺失监督误解释为一个伪造的 teacher 分布。
                    dist_ego_ctrlsim_teacher = torch.distributions.Categorical(
                        logits=ego_ctrlsim_action_logits_batch[ego_ctrlsim_valid_mask],
                    )
                    # student 分布也必须对齐到同一批有效样本，否则 teacher/student
                    # 的 KL 会混入未采样 timestep，破坏当前稀疏 KL cadence 的语义。
                    dist_ego_ctrlsim_student = torch.distributions.Categorical(
                        logits=dist_protagonist.logits[ego_ctrlsim_valid_mask],
                    )
                    ego_ctrlsim_kl_div = torch.distributions.kl.kl_divergence(
                        dist_ego_ctrlsim_teacher,
                        dist_ego_ctrlsim_student,
                    )
                    ego_ctrlsim_kl_loss = ego_ctrlsim_kl_div.mean()  # mean KL loss over the valid subset of the mini-batch

                self.optimizer.zero_grad()  # clear the existing gradients before backpropagation

                # 在标准 PPO loss 上叠加两个互相独立的 KL 正则项
                # Add the two independent KL regularizers on top of the standard PPO loss.
                loss = (value_loss*self.value_loss_coef + action_loss - dist_entropy*self.entropy_coef)
                if use_model_kl_loss:
                    loss += (self.kl_loss_coef*kl_loss)
                if use_ego_ctrlsim_kl_loss:
                    loss += (self.kl_loss_coef * ego_ctrlsim_kl_loss)

                loss.backward()  # compute the gradients of the combined loss with respect to the model parameters

                if self.log_grad_norm:
                    grad_norms.append(self._grad_norm())

                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                            self.max_grad_norm)
                    
                if not discard_grad:
                    self.optimizer.step()  # update the model parameters using the computed gradients
                                
                value_loss_epoch += value_loss.item()  # Convert a tensor containing a single element (a scalar tensor) into a Python scalar (float` or `int`).
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                if use_model_kl_loss:
                    kl_loss_epoch += kl_loss.item()
                if use_ego_ctrlsim_kl_loss:
                    ego_ctrlsim_kl_loss_epoch += ego_ctrlsim_kl_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        if use_model_kl_loss:
            kl_loss_epoch /= num_updates
    
        # 即使某些 batch 没启用 ego_ctrlsim KL，也统一按 num_updates 做 epoch 平均
        # Average ego_ctrlsim KL over the full update count for consistent epoch-level logging.
        ego_ctrlsim_kl_loss_epoch /= num_updates

        info = {}
        if self.log_grad_norm:
            info = {'grad_norms': grad_norms}
        if use_model_kl_loss:
            info['kl_loss'] = kl_loss_epoch
        # 单独记录 ego_ctrlsim KL，避免和旧的在线 KL 指标混淆。
        # Log ego_ctrlsim KL separately so it does not get mixed with the legacy online KL metric.
        if ego_ctrlsim_kl_loss_epoch > 0.0:
            info['ego_ctrlsim_kl_loss'] = ego_ctrlsim_kl_loss_epoch

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, info
