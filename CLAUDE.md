# heIPyDASA - Claude Code Project Guide

## Project Overview

PyDASA is a Python library for dimensional analysis using the Buckingham Pi theorem. It targets scientific and engineering domains. The library is in **Alpha** (v0.7.0+), published on PyPI, with a single maintainer.

- **Source layout**: `src/pydasa/` (standard src-layout)
- **Tests**: `tests/pydasa/` (pytest), `tests/notebooks/` (nbval)
- **Docs**: `docs/source/` (Sphinx + ReadTheDocs)
- **CI/CD**: GitHub Actions (`.github/workflows/test.yml`, `.github/workflows/release.yml`)
- **Config**: JSON files in `src/pydasa/core/cfg/` are the **source of truth** for naming and data
- **venv**: Always active. Located at `venv/` in project root

## Virtual Environment

**Before running ANY Python command** (pytest, pip, python, semantic-release, sphinx, etc.), verify the venv is active:

```bash
echo $VIRTUAL_ENV
```

- If the output contains this project's path, proceed.
- If empty or pointing elsewhere, activate it first: `source venv/Scripts/activate` (Windows/Git Bash)
- NEVER run pytest, pip install, or python against the system Python. Always through the venv.

## Development Workflow

New work follows these stages (usually on `dev` or a feature branch):

```
proof-of-concept -> planning -> dev -> test -> analysis -> refinement -> merge to main -> publish
```

| Stage                      | Where                       | What happens                                                |
| -------------------------- | --------------------------- | ----------------------------------------------------------- |
| **proof-of-concept** | `lab/<name>/`             | 1-2 .py + 1 file, max ~500 lines. Test a hypothesis quickly |
| **planning**         | `notes/devlog.md`         | Tech review, decide scope, document approach                |
| **dev**              | `src/pydasa/`, `tests/` | Implement as module code with tests                         |
| **test**             | CI / local pytest           | Verify correctness, coverage >=90%                          |
| **analysis**         | notebooks or `lab/`       | Validate behavior with real data or edge cases              |
| **refinement**       | `src/pydasa/`             | Clean up based on analysis findings                         |
| **merge + publish**  | `main`                    | PR to main triggers semantic-release and PyPI publish       |

Not everything graduates. Dead ends are logged in `notes/devlog.md` with reasoning.

## Lab - Proof of Concepts

`lab/` is a disposable scratchpad for testing ideas:

- Each PoC is a folder: `lab/<name>/`
- Max 2 `.py` files + 1 supporting file per PoC
- Short code (~500 lines total)
- NOT packaged, NOT tested in CI
- Graduates into `src/pydasa/` or stays as a dead end documented in devlog

## Development Log

`notes/devlog.md` is an experiment journal — NOT a changelog. It tracks:

- What was tried, accepted, or abandoned
- Dead ends and why they were dead ends
- Time-stamped entries with stage tags
- Decisions and their reasoning

## Branching and Releases

- **main**: production branch, triggers semantic-release and PyPI publish
- **dev**: development/integration branch, where most new work happens
- Feature branches merge into `dev`, then `dev` merges into `main`
- Releases are fully automated via `python-semantic-release`
- Version tracked in `src/pydasa/_version.py` and `pyproject.toml`
- Tag format: `v{version}` (e.g., `v0.7.0`)

## Commit Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>
```

| Type         | When to use                        | Bump  |
| ------------ | ---------------------------------- | ----- |
| `feat`     | New feature                        | minor |
| `fix`      | Bug fix                            | patch |
| `perf`     | Performance improvement            | patch |
| `docs`     | Documentation only                 |       |
| `style`    | Formatting, no logic change        |       |
| `refactor` | Code change, no new feature or fix |       |
| `test`     | Adding or updating tests           |       |
| `build`    | Build system or dependencies       |       |
| `ci`       | CI configuration                   |       |
| `chore`    | Maintenance tasks                  |       |

## Coding Conventions

- Python 3.10+ only
- No excessive column-alignment spacing
- No inline command chaining; break into sequential steps with intermediate variables
- Type hints on all function signatures
- Docstrings with Args/Returns format
- Local variables inside functions/methods: prefix with `_` (e.g., `_lam`, `_cfg`)
- Variable names use acronyms to avoid search collisions (e.g., `_mu` not `_service_rate`)
- Function names: verb-first, then `_` and up to 5 acronyms (e.g., `compute_dc_theta`, `plot_yoly_2d`)
- JSON config files in `core/cfg/` are authoritative over code comments or notebook markdown

## Testing

```bash
# Run full test suite with coverage
pytest tests/ -v --tb=short --cov=src/pydasa --cov-report=term

# Run a specific module's tests
pytest tests/pydasa/dimensional/ -v

# Run notebook tests (non-blocking in CI)
pytest --nbval-lax --nbval-current-env tests/notebooks/
```

- Target: 90%+ coverage per module
- CI matrix: Python 3.10, 3.11, 3.12 on Ubuntu
- Coverage uploaded to Codecov

## Building and Publishing

```bash
# Install in dev mode
pip install -e ".[dev]"

# Build package
python -m build

# Semantic release (automated in CI, manual only if needed)
semantic-release version
semantic-release publish
```

## Key Architecture

- **Entry point**: `AnalysisEngine` in `workflows/phenomena.py`
- **Domain objects**: `Variable`, `Coefficient`, `Dimension`, `Schema`, `Matrix`
- **Analysis**: `SensitivityAnalysis`, `MonteCarloSimulation`
- **I/O**: `load()` / `save()` in `core/io.py`
- **Validation**: decorator-based, in `validations/`
- **Data structures**: custom `ArrayList`, `SingleLinkedList`, `SCHashTable` in `structs/`

## Debugging Rules

- Read the actual code before suggesting fixes; never assume structure
- Verify dictionary keys, variable names, and data types before editing
- Identify root cause before proposing a fix
- Watch for: key format mismatches, deep vs shallow copy, dtype issues in pandas

## Research Skills

Domain-specific skills for data analysis, writing, and documentation are in `.claude/skills/`. These are separate from DevOps commands in `.claude/commands/`.
