import numpy as np
import torch


def sample_paths(cfg, bs, N=None):
    N = cfg.N if N is None else N
    dt = cfg.T / N
    dW = np.sqrt(dt) * np.random.randn(bs, N, cfg.d)
    log_incr = (cfg.r - cfg.q - 0.5 * cfg.sigma**2) * dt + cfg.sigma * dW
    X = np.empty((bs, N + 1, cfg.d))
    X[:, 0] = cfg.S0
    X[:, 1:] = cfg.S0 * np.exp(np.cumsum(log_incr, axis=1))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(dW, dtype=torch.float32)


def g_aon(x, K):
    # Asset-or-nothing payoff on equally weighted basket
    B = x.mean(-1)
    return B * (B > K).to(x.dtype)


def g_con(x, K, Q):
    # Cash-or-nothing payoff on equally weighted basket
    B = x.mean(-1)
    return Q * (B > K).to(x.dtype)


def geo_mean(x):
    return torch.exp(torch.log(x).mean(-1))


def g_geo_aon(x, K):
    # Asset-or-nothing payoff on geometric basket,
    # log space to avoid numerical overflow
    G = geo_mean(x)
    return G * (G > K).to(x.dtype)


PAYOFFS = {"aon": g_aon, "geo": g_geo_aon}


def geo_params(cfg):
    sigma_G = cfg.sigma / np.sqrt(cfg.d)
    q_G = cfg.q + 0.5 * cfg.sigma**2 - 0.5 * sigma_G**2
    return sigma_G, q_G


def geo_d1(t, G, cfg):
    sigma_G, q_G = geo_params(cfg)
    tau = cfg.T - t
    return (torch.log(G / cfg.K) + (cfg.r - q_G + 0.5 * sigma_G**2) * tau) / (sigma_G * np.sqrt(tau))


def geo_price(t, G, cfg):
    Phi = torch.distributions.Normal(0.0, 1.0).cdf
    _, q_G = geo_params(cfg)
    return G * np.exp(-q_G * (cfg.T - t)) * Phi(geo_d1(t, G, cfg))


def geo_Z(t, x, cfg):
    Phi = torch.distributions.Normal(0.0, 1.0).cdf
    phi = lambda z: torch.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    sigma_G, q_G = geo_params(cfg)
    G = geo_mean(x)
    tau = cfg.T - t
    d1 = geo_d1(t, G, cfg)
    V_G = np.exp(-q_G * tau) * (Phi(d1) + phi(d1) / (sigma_G * np.sqrt(tau)))
    return (cfg.sigma / cfg.d * G * V_G).unsqueeze(-1).expand(-1, cfg.d)


def geo_exact_t0(cfg):
    x0 = torch.tensor(cfg.S0, dtype=torch.float32).unsqueeze(0)
    y0 = geo_price(0.0, geo_mean(x0), cfg).item()
    z0 = geo_Z(0.0, x0, cfg)[0]
    return y0, z0
