export MPLBACKEND := "Agg"

# Show development commands.
default:
    @just --list

# Install the locked development environment.
setup:
    uv sync --locked

# Install Git hooks (run once per checkout).
hooks:
    uv run --locked prek install

# Apply safe lint fixes and format Python and Markdown.
fmt:
    uv run --locked ruff check --fix
    uv run --locked ruff format

# Run file checks, Ruff, and the full-project type check.
lint:
    uv run --locked prek run --all-files

# Run tests, doctests, and the 90% coverage gate.
test *args:
    uv run --locked pytest {{ args }}

# Build the Zensical site.
docs-build:
    uv run --locked --no-default-groups --group docs zensical build --strict

# Preview the Zensical site.
docs *args:
    uv run --locked --no-default-groups --group docs zensical serve {{ args }}

# Build a wheel and source distribution.
build:
    uv build --no-sources

# Run all checks used before a release.
check: lint test docs-build build
