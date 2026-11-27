import torch
from torch import nn


def subnet(d_in, d_out, width, h, act=nn.ELU, bn=False):
    layers = []
    in_size = d_in
    for _ in range(h):
        layers.append(nn.Linear(in_features=in_size, out_features=width))
        if bn:
            layers.append(nn.BatchNorm1d(width))
        layers.append(act())
        in_size = width
    layers.append(nn.Linear(in_features=width, out_features=d_out))
    return nn.Sequential(*layers)


class SubnetC(nn.Module):
    def __init__(self, d_in, d_out, width, h):
        super().__init__()
        self.hidden = nn.ModuleList()
        in_size = d_in
        for _ in range(h):
            self.hidden.append(nn.Linear(in_features=in_size, out_features=width))
            in_size = width
        self.out = nn.Linear(in_features=width, out_features=d_out)
        self.act = nn.ELU()

    def forward(self, x):
        a = self.act(self.hidden[0](x))
        skip = a
        for k in range(2, len(self.hidden) + 1):
            a = self.act(self.hidden[k-1](a))
            if k % 2 == 1:
                a = a + skip
                skip = a
            elif k == len(self.hidden):
                a = a + skip
        return self.out(a)


def make_subnet(net_type, d_in, d_out, width, h):
    if net_type == "A":
        return subnet(d_in, d_out, width, h, act=nn.ReLU, bn=True)
    if net_type == "B":
        return subnet(d_in, d_out, width, h, act=nn.ELU, bn=False)
    if net_type == "C":
        return SubnetC(d_in, d_out, width, h)
    raise ValueError(f"unknown net_type {net_type}")


class Model(nn.Module):
    def __init__(self, cfg, net_type, width, h, g_fn=None):
        super().__init__()
        self.z0 = torch.nn.Parameter(torch.zeros(cfg.d))
        self.y0 = torch.nn.Parameter(torch.tensor(0.0))
        self.r = cfg.r
        self.dt = cfg.dt
        self.g_fn = g_fn
        self.net_type = net_type
        d_in = cfg.d + 1 if g_fn is not None else cfg.d
        self.nets = nn.ModuleList([make_subnet(net_type, d_in, cfg.d, width, h) for _ in range(cfg.N - 1)])

    def z_at(self, step, x_n):
        if self.g_fn is not None:
            x_n = torch.cat([x_n, self.g_fn(x_n).unsqueeze(-1)], dim=-1)
        return self.nets[step - 1](x_n)

    def forward(self, X, dW):
        Y = torch.ones(len(X)) * self.y0
        Y = (1 + self.r * self.dt) * Y + (self.z0 * dW[:, 0, :]).sum(-1)
        for step in range(1, len(self.nets) + 1):
            Z = self.z_at(step, X[:, step, :])
            Y = (1 + self.r * self.dt) * Y + (Z * dW[:, step, :]).sum(-1)
        return Y
