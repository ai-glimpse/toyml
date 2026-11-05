# Contributing

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[just](https://just.systems/man/en/packages.html). Use the uv version required by
`pyproject.toml`; `.python-version` selects the development Python version.

```sh
just setup
just hooks
```

## Development

```sh
just fmt                 # Apply safe Ruff fixes and formatting
just lint                # Run prek, Ruff, and ty
just test                # Run tests, doctests, and the coverage gate
just test -k kmeans      # Pass arguments to pytest
just build               # Build the wheel and source distribution
just verify-dist         # Verify packaging and isolated installation
just check               # Run all checks
```

Commit `uv.lock` with dependency changes. Use `uv lock --upgrade` to update dependencies;
project commands and CI use `--locked`.

Library code lives in `src/toyml/`. Run examples and scripts through the installed package:

```sh
uv run --locked python examples/clustering/kmeans.py
```

When changing Python compatibility or dependencies, test the supported Python versions
listed in `pyproject.toml`:

```sh
MPLBACKEND=Agg uv run --locked --python 3.15 pytest
```

## Model and test conventions

- Constructors accept configuration; fitted/internal dataclass fields use `init=False`.
- Fitting preserves configuration and replaces fitted state.
- Methods preserve caller inputs. Copy containers according to their structure;
  reserve `deepcopy` for complex mutable object graphs that require recursive isolation.
- Returned label lists are independent containers. Public fitted attributes remain mutable.
- Inference before fitting raises `ValueError`.
- Integer seeds reproduce fits within the same implementation and environment.
  Instances own their random streams; `None` does not promise reproducibility.

Use pytest and Hypothesis. Compare cluster partitions rather than arbitrary label numbers.
Keep shared contracts in `tests/test_model_ownership.py` and
`tests/test_random_reproducibility.py`, and model-specific properties with the model's tests.
Deferred bugs live in `tests/test_known_issues.py` with strict xfail markers; replace them
with passing regression tests when fixed.

## Documentation

Edit algorithm guides in `docs/`; API references are generated from source docstrings.
Zensical configuration lives in `zensical.toml`.

```sh
just docs-build
just docs
```

Run edited examples. For layout or configuration changes, check navigation, search,
formulas, images, and narrow-screen rendering in the preview.

Regenerate cover assets with:

```sh
uv run --locked python scripts/generate_cover.py
```

Use `--points 1000 --output-root .cache/cover-preview` for a quick preview.

## Releases

Choose an unpublished version with `uv version`, commit the version and lockfile,
and run `just check`. Publish a GitHub Release with a matching version tag
(an optional `v` prefix is accepted) and user-facing release notes.
The release workflow publishes the package to PyPI using trusted publishing.
