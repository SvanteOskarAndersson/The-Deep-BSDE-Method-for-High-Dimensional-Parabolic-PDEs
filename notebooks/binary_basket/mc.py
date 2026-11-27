import numpy as np
import torch
import matplotlib.pyplot as plt

from config import save_plot
from payoffs import sample_paths


def mc_price(cfg, n_paths, payoff, batch_size=1000, N=1):
    payoffs = []
    for _ in range(n_paths // batch_size):
        X, _ = sample_paths(cfg, batch_size, N=N)
        payoffs.append(payoff(X[:, -1], cfg.K))
    payoffs = torch.cat(payoffs).double()
    disc = np.exp(-cfg.r * cfg.T)
    price = disc * payoffs.mean().item()
    std_err = disc * payoffs.std().item() / np.sqrt(len(payoffs))
    return price, std_err


def mc_convergence(cfg, payoff, paths, batch_size=1000):
    est = []
    err = []
    for path in paths:
        mc, se = mc_price(cfg, path, payoff, batch_size=batch_size)
        err.append(se)
        est.append(mc)
        print(f"MC estimation: u = {mc}, se = {se} with M = {path}")
    return est, err


def plot_mc_convergence(paths, est, err):
    plt.loglog(paths, err, label = "SE")
    plt.loglog(paths, err[0]*np.sqrt(paths[0]/np.array(paths)), '--', label='O(M^-1/2)')
    plt.legend()
    plt.xlabel('Num. of paths (M)')
    plt.ylabel('Standard Error')
    save_plot("mc_std_error")
    plt.show()

    plt.loglog(paths, est)
    plt.title('Option Value estimates')
    plt.xlabel('Num. of paths (M)')
    plt.ylabel('Value (V)')
    save_plot("mc_estimate")
    plt.show()
