import torch
from torch import nn
import numpy as np

# Codes adapted from the paper of Sitzmann et al (2020). https://arxiv.org/abs/2006.09661
# Introduction to the Siren model is available here : https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, is_second=False, omega_0=30, omega_0_t=5, learn_first=True):
        super().__init__()
        self.omega_0 = omega_0
        self.omega_0_t = omega_0_t
        self.is_first = is_first
        self.is_second = is_second
        self.out_features = out_features
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.learn_first = learn_first
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                nb_w = int(np.round(np.power(self.out_features, 1 / self.in_features)))
                w_x = torch.linspace(0, 1 / self.in_features, steps=nb_w)
                w_y = torch.linspace(-1 / self.in_features, 1 / self.in_features, steps=nb_w)
                w_t = torch.linspace(-self.omega_0_t / self.omega_0 / self.in_features, self.omega_0_t / self.omega_0 / self.in_features, steps=nb_w)
                grid = torch.meshgrid((w_t, w_x, w_y))
                self.linear.weight.data = torch.stack(grid, dim=-1).reshape(-1, 3)
                self.linear.weight.requires_grad = self.learn_first

            elif self.is_second:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=25, hidden_omega_0=25., omega_0_t=5, learn_first=True):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0,
                                  omega_0_t=omega_0_t, learn_first=learn_first))

        self.net.append(SineLayer(hidden_features, hidden_features,
                                  is_second=True, omega_0=hidden_omega_0))

        for i in range(hidden_layers - 1):
            self.net.append(SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords