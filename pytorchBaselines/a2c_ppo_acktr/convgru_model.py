import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from pytorchBaselines.a2c_ppo_acktr.distributions import (
    Bernoulli,
    Categorical,
    DiagGaussian,
)
from pytorchBaselines.a2c_ppo_acktr.utils import init
from torch import optim
from torch.optim import lr_scheduler


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        self.gru = nn.GRU(recurrent_input_size, hidden_size)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class ConvGRU(NNBase):
    def __init__(self, obs_space_dict, config):
        self.config = config
        self._is_recurrent = True

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        # NOTE: critic could be 4-8x wider than actor
        actor_hidden_size = 64
        critic_hidden_size = 256
        gru_input_size = config.ConvGRU.input_size
        # GRU output dims, also shared between actor and critic
        shared_ac_size = config.ConvGRU.hidden_size
        recurrent = True
        super(ConvGRU, self).__init__(recurrent, gru_input_size, shared_ac_size)

        self.conv_blk = nn.Sequential(
            init_(
                nn.Conv1d(
                    config.lidar.cfg["num_beams"],
                    out_channels=64,
                    kernel_size=7,
                    stride=2,
                )
            ),
            nn.LeakyReLU(),
            init_(nn.Conv1d(64, out_channels=128, kernel_size=5, stride=2)),
            nn.LeakyReLU(),
            init_(nn.Conv1d(128, out_channels=256, kernel_size=3, stride=1)),
        )

        self.actor = nn.Sequential(
            init_(nn.Linear(shared_ac_size, actor_hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(actor_hidden_size, actor_hidden_size)),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(shared_ac_size, critic_hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(critic_hidden_size, critic_hidden_size)),
            nn.Tanh(),
        )
        self.critic_linear = init_(nn.Linear(critic_hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x = self.conv_blk(x)
            print(x.shape)
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
