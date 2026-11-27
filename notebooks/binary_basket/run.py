import argparse
import numpy as np
import torch

from config import Config
from payoffs import PAYOFFS, geo_exact_t0
from mc import mc_convergence, plot_mc_convergence
from nets import Model
from training import train, eval_loss, eval_Z, plot_histories, plot_z_error, print_compare_table, plot_compare_z_error
import time

def run_mc(cfg, payoff_name):
    paths = [1000, 10_000, 100_000, 1_000_000]
    est, err = mc_convergence(cfg, PAYOFFS[payoff_name], paths)
    if payoff_name == "geo":
        print(f"exact: u = {geo_exact_t0(cfg)[0]}")
    plot_mc_convergence(paths, est, err)


def run_train(cfg, payoff_name, net_types, n_iter, h):
    payoff = PAYOFFS[payoff_name]
    histories = {}
    errors = {}
    for net_type in net_types:
        torch.manual_seed(0)
        np.random.seed(0)

        g_fn = (lambda x: payoff(x, cfg.K)) if net_type == "C" else None
        model = Model(cfg, net_type, width=cfg.d + 10, h=h, g_fn=g_fn)
        n_params = sum(p.numel() for p in model.parameters())

        histories[net_type] = train(model, cfg, payoff, n_iter=n_iter)
        print(f"{net_type} ({n_params} params): y0 = {histories[net_type]['y0'][-100:].mean():.4f}, "
              f"loss = {histories[net_type]['loss'][-100:].mean():.1f}, "
              f"z0 = {model.z0.detach().numpy().round(2)}")

        if payoff_name == "geo":
            res = eval_Z(model, cfg)
            errors[net_type] = res["rel_err"]
            print(f"   exact y0 = {res['y0_exact']:.4f}, exact z0 = {res['z0_exact']:.3f}, "
                  f"z0 rel err = {res['z0_err']:.3f}")
            print(f"   loss net = {eval_loss(model, cfg, payoff):.1f}, loss exact Z = {res['loss_exact']:.1f}")

    plot_histories(histories, "loss", "loss")
    plot_histories(histories, "y0", "y0")
    if errors:
        plot_z_error(errors)

def run_compare(cfg, payoff_name, net_types, n_iter, h, n_seeds):
    payoff = PAYOFFS[payoff_name]
    results = {net_type: [] for net_type in net_types}

    for net_type in net_types:
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            g_fn = (lambda x: payoff(x, cfg.K)) if net_type == "C" else None
            model = Model(cfg, net_type, width=cfg.d + 10, h=h, g_fn=g_fn)

            start = time.perf_counter()
            history = train(model, cfg, payoff, n_iter=n_iter)
            elapsed = time.perf_counter() - start

            run = {"y0": history["y0"][-100:].mean(),
                   "loss": eval_loss(model, cfg, payoff, seed=12345),
                   "time": elapsed}
            if payoff_name == "geo":
                res = eval_Z(model, cfg)
                run["y0_err"] = abs(run["y0"] - res["y0_exact"])
                run["z0_err"] = res["z0_err"]
                run["Z_err"] = res["rel_err"].mean()
                run["ratio"] = run["loss"] / res["loss_exact"]
                run["rel_err"] = res["rel_err"]
            results[net_type].append(run)
            print(f"{net_type} seed {seed}: y0 = {run['y0']:.4f}, "
                  f"loss = {run['loss']:.1f}, time = {elapsed:.1f} s")

    print_compare_table(results)
    if payoff_name == "geo":
        plot_compare_z_error(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="+", choices=["mc", "train", "compare"])
    parser.add_argument("--model", nargs="+", choices=["A", "B", "C"], default=["B"])
    parser.add_argument("--payoff", choices=["aon", "geo"], default="aon")
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--K", type=float, default=100.0)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--h", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    cfg = Config(d=args.d, K=args.K)

    if "mc" in args.parts:
        run_mc(cfg, args.payoff)
    
    if "compare" in args.parts:
        run_compare(cfg, args.payoff, args.model, args.n_iter, args.h, args.seeds)

    if "train" in args.parts:
        run_train(cfg, args.payoff, args.model, args.n_iter, args.h)
