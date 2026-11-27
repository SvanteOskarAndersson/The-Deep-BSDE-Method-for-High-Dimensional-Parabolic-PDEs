from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR = Path(__file__).parent / "plots"


def save_plot(name):
    PLOT_DIR.mkdir(exist_ok=True)
    plt.savefig(PLOT_DIR / f"{name}.png", dpi=200, bbox_inches="tight")


@dataclass
class Config:
    d: int = 10
    T: float = 1.0
    N: int = 20
    r: float = 0.05
    q: float = 0.0
    sigma: float = 0.2
    K: float = 100.0
    s0: float = 100.0

    @property
    def dt(self):
        return self.T / self.N

    @property
    def S0(self):
        return self.s0 * np.ones(self.d)
