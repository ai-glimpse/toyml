# Contributing

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[just](https://just.systems/man/en/packages.html). Python 3.11 or later is supported;
`.python-version` selects Python 3.14 for development. uv downloads Python when needed.

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
checks the entire `toyml/`, `tests/`, and `examples/` trees. Keep deliberately invalid
inputs in runtime validation tests documented with specific `ty: ignore[...]` comments.

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

The Publish to PyPI workflow checks the release tag against `pyproject.toml`, builds the
package without installing development dependencies, and publishes using the
existing PyPI trusted publisher in the `pypi` environment.
