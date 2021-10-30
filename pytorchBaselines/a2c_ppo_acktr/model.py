import torch
import torch.nn as nn
from pytorchBaselines.a2c_ppo_acktr.convgru_model import ConvGRU
from pytorchBaselines.a2c_ppo_acktr.distributions import (
    Bernoulli,
    Categorical,
    DiagGaussian,
)
from pytorchBaselines.a2c_ppo_acktr.srnn_model import SRNN


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if base == "srnn":
            base = SRNN
            self.base = base(obs_shape, base_kwargs)
            self.srnn = True
        elif base == "convgru":
            base = ConvGRU
            self.base = base(obs_shape, base_kwargs)
            self.convgru = True
        else:
            raise NotImplementedError
        dist_input_sz = (
            self.base.actor.fc2.out_features if self.convgru else self.base.output_size
        )

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(dist_input_sz, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(dist_input_sz, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(dist_input_sz, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if not hasattr(self, "srnn"):
            self.srnn = False
        if self.srnn:
            value, actor_features, rnn_hxs = self.base(
                inputs, rnn_hxs, masks, infer=True
            )
        elif self.convgru:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        else:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):

        if self.srnn:
            value, _, _ = self.base(inputs, rnn_hxs, masks, infer=True)
        elif self.convgru:
            value, _, _ = self.base(inputs, rnn_hxs, masks)

        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
