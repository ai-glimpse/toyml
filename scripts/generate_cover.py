"""Render the Clifford-attractor cover as PDF, SVG, and a small web PNG."""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402


def clifford_attractor(
    a: float,
    b: float,
    c: float,
    d: float,
    n_points: int = 5_000_000,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Iterate the Clifford equations from the initial point (0.1, 0.1)."""
    if n_points < 2:
        msg = "n_points must be at least 2"
        raise ValueError(msg)
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    x[0], y[0] = 0.1, 0.1
    for i in range(n_points - 1):
        x[i + 1] = np.sin(a * y[i]) + c * np.cos(a * x[i])
        y[i + 1] = np.sin(b * x[i]) + d * np.cos(b * y[i])
    return x, y


def generate_cover(n_points: int = 5_000_000, output_root: Path | None = None) -> None:
    """Write print assets under assets/ and a 400px PNG under docs/assets/."""
    root = output_root if output_root is not None else Path(__file__).resolve().parent.parent
    x, y = clifford_attractor(-1.3, -1.3, -1.8, -1.9, n_points)
    histogram, _, _ = np.histogram2d(x, y, bins=1200)
    colors = ["#ffffff00", "#fff5e6", "#d4af37", "#008080", "#002b36"]
    cmap = LinearSegmentedColormap.from_list("latent_space", colors)
    figure = plt.figure(figsize=(10, 10))
    try:
        figure.patch.set_alpha(0)
        axes = figure.add_axes((0, 0, 1, 1))
        axes.set_axis_off()
        axes.patch.set_alpha(0)
        axes.imshow(
            np.log1p(histogram).T,
            origin="lower",
            cmap=cmap,
            interpolation="bicubic",
            extent=(-2.5, 2.5, -2.5, 2.5),
        )
        # The PNG is displayed at 200px; 400px also supports a 2x display.
        outputs = [
            (root / "assets/cover.pdf", 300),
            (root / "assets/cover.svg", 100),
            (root / "docs/assets/cover.png", 40),
        ]
        for path, dpi in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0)
            print(f"Generated {path}")
    finally:
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=5_000_000, help="number of attractor points (default: 5000000)")
    parser.add_argument("--output-root", type=Path, help="output root (default: repository root)")
    arguments = parser.parse_args()
    generate_cover(n_points=arguments.points, output_root=arguments.output_root)
