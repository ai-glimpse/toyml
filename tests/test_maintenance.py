import math
import runpy
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_kmeans_example_uses_installed_package() -> None:
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "examples/clustering/kmeans.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Training labels:" in result.stdout
    assert "Centroids:" in result.stdout
    assert "Prediction for [1.5, 1.5]:" in result.stdout


def test_clifford_equations() -> None:
    attractor = runpy.run_path(str(ROOT / "scripts/generate_cover.py"))["clifford_attractor"]
    x, y = attractor(-1.3, -1.3, -1.8, -1.9, n_points=3)
    assert x[0] == y[0] == 0.1
    for i in range(2):
        assert x[i + 1] == pytest.approx(math.sin(-1.3 * y[i]) - 1.8 * math.cos(-1.3 * x[i]))
        assert y[i + 1] == pytest.approx(math.sin(-1.3 * x[i]) - 1.9 * math.cos(-1.3 * y[i]))


@pytest.mark.parametrize("points", [-1, 0, 1])
def test_cover_rejects_too_few_points(points: int) -> None:
    attractor = runpy.run_path(str(ROOT / "scripts/generate_cover.py"))["clifford_attractor"]
    with pytest.raises(ValueError, match="n_points must be at least 2"):
        attractor(-1.3, -1.3, -1.8, -1.9, n_points=points)


def test_cover_cli_writes_expected_assets(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "scripts/generate_cover.py"),
            "--points",
            "1000",
            "--output-root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        check=True,
    )
    assert (tmp_path / "assets/cover.pdf").read_bytes().startswith(b"%PDF-")
    assert "<svg" in (tmp_path / "assets/cover.svg").read_text()
    png = (tmp_path / "docs/assets/cover.png").read_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", png[16:24]) == (400, 400)
