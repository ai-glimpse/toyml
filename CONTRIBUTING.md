# Contributing

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[just](https://just.systems/man/en/packages.html). Python 3.11 or later is supported;
`.python-version` selects Python 3.14 for development. uv downloads Python when needed.

`[tool.uv].required-version` in `pyproject.toml` pins the uv version for local commands
and all GitHub Actions workflows. Install that version (for example,
`uv self update 0.13.7` for a standalone installation) before running project commands.
When updating uv, change the constraint once and validate the lockfile and builds.

```sh
just setup
just hooks
```

`uv.lock` records the development, test, and documentation environments. Commit it
with dependency changes. Run `uv lock --upgrade` deliberately to update dependencies;
CI uses `--locked` and rejects an outdated lockfile.

Pull requests run two jobs: [Python CI](.github/workflows/ci.yml) runs checks/tests/packaging
on the development Python version from `.python-version`, and
[Documentation](.github/workflows/docs.yml) runs a strict Zensical build. When changing compatibility
or dependencies, use `--python` to repeat the tests locally for Python 3.11–3.15:

```sh
MPLBACKEND=Agg uv run --locked --python 3.11 pytest
```

## Development

```sh
just fmt                 # Apply safe Ruff fixes and formatting
just lint                # Run prek file checks, Ruff, and ty
just test                # Run tests, doctests, and the 90% coverage gate
just test -k kmeans      # Pass arguments to pytest
just build               # Build the wheel and source distribution
just verify-dist         # Check metadata, sdist rebuild, and isolated wheel installation
just check               # Run lint, tests, docs, and packaging
```

The default `dev` dependency group includes `lint`, `test`, and prek. The test group
includes plotting dependencies because the suite exercises plotting. `just` selects
the non-interactive Matplotlib backend, so tests also run without a display.

For isolated jobs, select only the required group:

```sh
uv run --locked --no-default-groups --group test pytest
uv run --locked --no-default-groups --group docs zensical build --strict
```

`prek.toml` is the native hook configuration. Basic file checks use prek's built-in
hooks, without fetching a separate hook repository or installing Python hook environments.
Ruff and ty run from the locked project environment in both Git hooks and CI. ty
checks the entire `toyml/`, `tests/`, `examples/`, and `scripts/` trees. Keep deliberately invalid
inputs in runtime validation tests documented with specific `ty: ignore[...]` comments.

`.editorconfig` defines UTF-8, LF, final newlines, and space indentation (four spaces,
or two for YAML). `.gitattributes` keeps text checkouts on LF and marks binary assets.

## Documentation

Zensical reads `zensical.toml` and generates API documentation from the source using
mkdocstrings-python. It does not require Pandoc or Git history/API access.

```sh
just docs-build
just docs
```

The site is built into `site/`. When changing documentation configuration, check the
home page, algorithm API pages, AGNES image, K-means formulas, search, light/dark
modes, and the configured `/toyml/` URL prefix.

Algorithm pages follow **Algorithm → Inputs and limitations → Minimal example → API**.
Run the examples when changing them and document the current implementation's limits.
The 400 × 400 cover in `docs/assets/cover.png` is displayed at 200 × 200; the documentation
serves it locally, while the README uses its GitHub raw URL for GitHub/PyPI rendering.
`assets/cover.svg` is the full-resolution source. Regenerate the cover with
`uv run --locked python scripts/generate_cover.py`; PDF/SVG outputs stay in `assets/`.

The Documentation workflow builds once and uploads a GitHub Pages artifact. Pull requests
only build; pushes to `master` deploy the same artifact. The repository's Pages
source must be set to **GitHub Actions** in Settings → Pages.

## Packaging and releases

The core library has no runtime dependencies. Users who need plots can install
`toyml[plot]`; documentation tools and testing dependencies are not shipped as
runtime requirements.

`uv build --no-sources` builds the wheel and source distribution with uv_build. Keep
`toyml/py.typed`, the license, and all Python modules in the wheel. Test an installed
wheel from outside the checkout to avoid accidentally importing the working tree.

`just verify-dist` builds fresh artifacts in a temporary directory, checks metadata,
all Python modules, `py.typed`, and the license, then compares the wheel with a rebuild
from the sdist. It installs the wheel without dependencies into a fresh virtual
environment outside the checkout, imports every module, and runs a clustering example.
`just check` includes this verification; CI keeps its existing build step.

Maintain release notes in [GitHub Releases](https://github.com/ai-glimpse/toyml/releases).
When preparing a release, generate notes from the merged pull requests and edit them
to highlight user-facing changes and migration steps.

Before publishing, choose a new version: `0.4.0` is already published and cannot be reused.
For the tooling migration, a minor bump (`uv version --bump minor`) would prepare `0.5.0`.
Commit the version change and refreshed `uv.lock`, run `just check`, and create a GitHub
Release with a matching tag (for example, `0.5.0` or `v0.5.0`). Its notes should explain
the Python 3.11 minimum, the `toyml[plot]` extra, and the uv/prek/Zensical development commands.
Publishing the GitHub Release triggers the PyPI workflow.

The Publish to PyPI workflow checks the release tag against `pyproject.toml`, builds the
package without installing development dependencies, and publishes using the
existing PyPI trusted publisher in the `pypi` environment.
