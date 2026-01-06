from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

    # Setup the figure with transparent background
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.patch.set_alpha(0)

    # Custom colormap: Transparent -> Golden -> Deep Teal -> Black (Modern Scientific)
    colors = ["#ffffff00", "#fff5e6", "#d4af37", "#008080", "#002b36"]
    cmap = LinearSegmentedColormap.from_list("latent_space", colors)

    # 2D Histogram with higher resolution
    print("Rendering density map...")
    nbins = 1200
    H, xedges, yedges = np.histogram2d(x, y, bins=nbins)
    H = np.log(H + 1)  # Log scale

    # Render with bicubic interpolation for smoothness
    ax.imshow(H.T, origin="lower", cmap=cmap, interpolation="bicubic", extent=[-2.5, 2.5, -2.5, 2.5])

    # Ensure assets directory exists
    assets_dir = Path(__file__).parent.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    # PDF for LaTeX (vector format - scalable)
    pdf_path = Path(__file__).parent.parent.parent / "figures/general/cover.pdf"
    # SVG for web (vector format - scalable)
    svg_path = assets_dir / "cover.svg"
    # PNG for general use (raster - use high DPI for quality)
    png_path = assets_dir / "cover.png"

    # Save as PDF (vector, ideal for LaTeX/print)
    plt.savefig(pdf_path, format="pdf", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"PDF cover generated at {pdf_path}")

    # Save as SVG (vector, ideal for web/scaling)
    plt.savefig(svg_path, format="svg", transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"SVG cover generated at {svg_path}")

    # Save as PNG (raster, high DPI for quality when scaling)
    plt.savefig(png_path, format="png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"PNG cover generated at {png_path}")


if __name__ == "__main__":
    generate_cover()
