import numpy as np
import torch
import matplotlib.pyplot as plt

from config import save_plot
from payoffs import sample_paths, g_geo_aon, geo_Z, geo_exact_t0


def make_optimizer(model, lr_y0=1.0, lr_z0=1.0, lr_nets=1e-2):
    # two different learningrates for y0 and Z
    opt = torch.optim.Adam([
        {"params": [model.y0], "lr": lr_y0},
        {"params": [model.z0], "lr": lr_z0},
        {"params": [p for name, p in model.named_parameters() if name not in ("y0", "z0")], "lr": lr_nets},
    ])
    return opt


def train(model, cfg, payoff, n_iter=1000, M=256, lr_y0=1.0, lr_z0=1.0, lr_nets=1e-2):
    loss_history = np.zeros(n_iter)
    y0_history = np.zeros(n_iter)
    opt = make_optimizer(model, lr_y0, lr_z0, lr_nets)

    for it in range(n_iter):
        X, dW = sample_paths(cfg, M)
        y = model(X, dW)

        #l2 loss:
        loss = torch.mean((y - payoff(X[:, -1, :], cfg.K))**2)

        opt.zero_grad() #clear prev. gradient
        loss.backward() #compute new gradient
        opt.step() # update y0, z0 and the weight

        loss_history[it] = loss.item()
        y0_history[it] = model.y0.item()

    return {"loss": loss_history, "y0": y0_history}

def eval_loss(model, cfg, payoff, M=5000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    model.eval()
    X, dW = sample_paths(cfg, M)
    with torch.no_grad():
        loss = torch.mean((model(X, dW) - payoff(X[:, -1, :], cfg.K)) ** 2).item()
    model.train()
    return loss

def eval_Z(model, cfg, M=5000):
    model.eval()
    X, dW = sample_paths(cfg, M)
    rel_err = np.zeros(cfg.N - 1)
    a = 1 + cfg.r * cfg.dt

    with torch.no_grad():
        y0_exact, z0_exact = geo_exact_t0(cfg)
        Y_ex = a * torch.ones(M) * y0_exact + (z0_exact * dW[:, 0, :]).sum(-1)
        for n in range(1, cfg.N):
            x_n = X[:, n, :]
            Z_net = model.z_at(n, x_n)
            Z_ex = geo_Z(n * cfg.dt, x_n, cfg)
            rel_err[n - 1] = ((Z_net - Z_ex).norm(dim=-1).mean() / Z_ex.norm(dim=-1).mean()).item()
            Y_ex = a * Y_ex + (Z_ex * dW[:, n, :]).sum(-1)

        g = g_geo_aon(X[:, -1, :], cfg.K)
        loss_exact = torch.mean((Y_ex - g) ** 2).item()
        z0_err = ((model.z0 - z0_exact).norm() / z0_exact.norm()).item()

    model.train()
    return {"rel_err": rel_err, "y0_exact": y0_exact, "z0_exact": z0_exact[0].item(),
            "z0_err": z0_err, "loss_exact": loss_exact}


def plot_histories(histories, key, ylabel):
    for net_type, history in histories.items():
        plt.plot(np.arange(len(history[key])), history[key], label = net_type)
    plt.xlabel("iterations")
    plt.ylabel(ylabel)
    plt.legend()
    save_plot(f"{key}_history")
    plt.show()


def plot_z_error(errors):
    for net_type, err in errors.items():
        plt.plot(np.arange(1, len(err) + 1), err, "o-", label = net_type)
    plt.xlabel("time step n")
    plt.ylabel("relative error in Z")
    plt.legend()
    save_plot("z_error")
    plt.show()

def print_compare_table(results):
    keys = [k for k in next(iter(results.values()))[0] if k != "rel_err"]
    print(f"{'net':<5}" + "".join(f"{k:>20}" for k in keys))
    for net_type, runs in results.items():
        row = f"{net_type:<5}"
        for k in keys:
            vals = np.array([run[k] for run in runs])
            row += f"{vals.mean():>11.3f} ± {vals.std():<6.3f}"
        print(row)


def plot_compare_z_error(results):
    for net_type, runs in results.items():
        E = np.array([run["rel_err"] for run in runs])
        n = np.arange(1, E.shape[1] + 1)
        plt.plot(n, E.mean(0), "o-", label = net_type)
        plt.fill_between(n, E.mean(0) - E.std(0), E.mean(0) + E.std(0), alpha=0.2)
    plt.xlabel("time step n")
    plt.ylabel("relative error in Z")
    plt.legend()
    save_plot("z_error_compare")
    plt.show()
