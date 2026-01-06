from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def clifford_attractor(a, b, c, d, n_points=5000000):
    # Initialize arrays
    x = np.zeros(n_points)
    y = np.zeros(n_points)

    # Starting point
    x[0], y[0] = 0.1, 0.1

    # Clifford Attractor equations:
    # x_{n+1} = sin(a * y_n) + c * cos(a * x_n)
    # y_{n+1} = sin(b * x_n) + d * cos(b * y_n)

    # We use a compiled loop or numpy vectorization trick for speed
    # But for millions of points in pure python, simple loop is slow.
    # Let's use a smaller N or a numba/cython if available, but here standard numpy.
    # To keep it standard-lib friendly, we iterate but optimize simply.

    # Pre-generate random noise to simulate "high dimensional latent search"
    # Actually, let's just implement the loop efficiently.

    cur_x, cur_y = x[0], y[0]

    # Using a slightly faster iteration by manual unraveling or just accepting 2-3s runtime
    for i in range(n_points - 1):
        x_next = np.sin(a * cur_y) + c * np.cos(a * cur_x)
        y_next = np.sin(b * cur_x) + d * np.cos(b * cur_y)
        x[i + 1] = x_next
        y[i + 1] = y_next
        cur_x, cur_y = x_next, y_next

    return x, y


def generate_cover():
    # Parameters for a more "Silk/Fabric" look, elegant and flowing
    a, b, c, d = -1.3, -1.3, -1.8, -1.9

    print("Generating attractor points...")
    n_points = 5000000

    x, y = clifford_attractor(a, b, c, d, n_points)

    # Setup the figure
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Custom colormap: White -> Golden -> Deep Teal -> Black (Modern Scientific)
    colors = ["#ffffff", "#fff5e6", "#d4af37", "#008080", "#002b36"]
    cmap = LinearSegmentedColormap.from_list("latent_space", colors)

    # 2D Histogram with higher resolution
    print("Rendering density map...")
    nbins = 1200
    H, xedges, yedges = np.histogram2d(x, y, bins=nbins)
    H = np.log(H + 1)  # Log scale

    # Render with bicubic interpolation for smoothness
    ax.imshow(H.T, origin="lower", cmap=cmap, interpolation="bicubic", extent=[-2.5, 2.5, -2.5, 2.5])

    # Output (relative to mlbook root, script is in toyml/scripts/)
    output_path = "../../figures/general/cover.pdf"
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    print(f"Cover generated at {output_path}")


if __name__ == "__main__":
    generate_cover()
