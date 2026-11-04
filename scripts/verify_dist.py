"""Build and verify release artifacts without importing the working tree."""

from __future__ import annotations

import email
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path


def wheel_contents(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as wheel:
        return {name: wheel.read(name) for name in wheel.namelist() if not name.endswith("/")}


def verify_wheel(path: Path, root: Path) -> dict[str, bytes]:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    contents = wheel_contents(path)
    metadata_path = next(name for name in contents if name.endswith(".dist-info/METADATA"))
    metadata = email.message_from_bytes(contents[metadata_path])
    assert metadata["Name"] == project["name"], "Package name mismatch"
    assert metadata["Version"] == project["version"], "Package version mismatch"
    assert metadata["Requires-Python"] == project["requires-python"], "Python requirement mismatch"
    assert metadata["License-Expression"] == project["license"], "License metadata mismatch"
    assert project["authors"][0]["email"] in metadata["Author-email"], "Author email mismatch"
    assert set(metadata.get_all("Project-URL", [])) == {f"{name}, {url}" for name, url in project["urls"].items()}, (
        "Project URL mismatch"
    )
    assert metadata.get_all("Provides-Extra", []) == ["plot"], "Missing plotting extra"
    requirements = metadata.get_all("Requires-Dist", [])
    assert requirements, "Missing plotting requirements"
    assert all("extra == 'plot'" in item or 'extra == "plot"' in item for item in requirements), (
        "Core package must have no runtime requirements"
    )
    assert "toyml/py.typed" in contents, "Missing PEP 561 marker"
    license_path = metadata_path.removesuffix("METADATA") + "licenses/LICENSE"
    assert contents[license_path] == (root / "LICENSE").read_bytes(), "Missing or changed license"
    modules = list((root / "src/toyml").rglob("*.py"))
    assert modules, "Missing package sources under src/toyml"
    for module in modules:
        name = module.relative_to(root / "src").as_posix()
        assert contents.get(name) == module.read_bytes(), f"Missing or changed module: {name}"
    return contents


def verify_dist() -> None:
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="toyml-verify-dist-") as directory:
        temporary = Path(directory)
        artifacts = temporary / "dist"
        subprocess.run(["uv", "build", "--no-sources", "--out-dir", str(artifacts)], cwd=root, check=True)
        wheel = next(artifacts.glob("*.whl"))
        contents = verify_wheel(wheel, root)

        source_dir = temporary / "source"
        with tarfile.open(next(artifacts.glob("*.tar.gz"))) as source:
            source.extractall(source_dir, filter="data")
        source_root = next(source_dir.iterdir())
        rebuilt = temporary / "rebuilt"
        subprocess.run(
            ["uv", "build", "--no-sources", "--wheel", "--out-dir", str(rebuilt), str(source_root)],
            cwd=temporary,
            check=True,
        )
        assert contents == wheel_contents(next(rebuilt.glob("*.whl"))), "sdist rebuilt a different wheel"

        environment = temporary / "venv"
        subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], cwd=temporary, check=True)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
            cwd=temporary,
            check=True,
        )
        # -I ignores both the checkout's import path and PYTHONPATH.
        smoke_test = """
import importlib
import importlib.metadata
import pkgutil
import sys
from pathlib import Path

import toyml
from toyml.clustering import AGNES

assert Path(toyml.__file__).is_relative_to(Path(sys.prefix)), toyml.__file__
assert {dist.metadata['Name'] for dist in importlib.metadata.distributions()} == {'toyml'}
for module in pkgutil.walk_packages(toyml.__path__, 'toyml.'):
    importlib.import_module(module.name)
labels = AGNES(n_cluster=2).fit_predict([[0.0], [0.1], [10.0], [10.1]])
assert len(labels) == 4 and labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]
assert importlib.metadata.version('toyml') == sys.argv[1], 'Installed version mismatch'
print(f'Imported isolated installation: {toyml.__file__}')
"""
        project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
        subprocess.run([str(python), "-I", "-c", smoke_test, project["version"]], cwd=temporary, check=True)
        print(f"Verified {wheel.name}: metadata, modules, license, py.typed, sdist rebuild, and isolated installation.")


if __name__ == "__main__":
    verify_dist()
