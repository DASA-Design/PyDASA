# Contributing to PyDASA

## Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/) to automatically generate releases and changelogs.

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: A new feature (triggers MINOR version bump: 0.2.75 → 0.3.0)
- **fix**: A bug fix (triggers PATCH version bump: 0.2.75 → 0.2.76)
- **docs**: Documentation only changes
- **style**: Code style changes (formatting, missing semicolons, etc.)
- **refactor**: Code refactoring without feature changes
- **perf**: Performance improvements (triggers PATCH bump)
- **test**: Adding or updating tests
- **build**: Changes to build system or dependencies
- **ci**: Changes to CI configuration files
- **chore**: Other changes that don't modify src or test files

### Breaking Changes

Add `BREAKING CHANGE:` in the commit footer to trigger a MAJOR version bump (0.2.75 → 1.0.0):

```
feat(api): redesign dimensional analysis API

BREAKING CHANGE: The AnalysisEngine constructor signature has changed.
Users must now pass schema as a named parameter.
```

### Examples

```bash
# Feature (minor bump: 0.2.75 → 0.3.0)
git commit -m "feat(workflows): add new monte carlo simulation workflow"

# Bug fix (patch bump: 0.2.75 → 0.2.76)
git commit -m "fix(phenomena): resolve type error in schema assignment"

# Performance improvement (patch bump)
git commit -m "perf(buckingham): optimize matrix calculations"

# Documentation (no version bump)
git commit -m "docs(readme): update installation instructions"

# Breaking change (major bump: 0.2.75 → 1.0.0)
git commit -m "feat(core)!: redesign Variable API

BREAKING CHANGE: Variable.value is now Variable.magnitude"
```

## Release Process

1. Make changes on `dev` branch
2. Create PR to `main` with conventional commit messages
3. Merge PR to `main`
4. GitHub Actions automatically:
   - Analyzes commit messages
   - Bumps version based on commit types
   - Updates `_version.py` and `pyproject.toml`
   - Creates GitHub release with changelog
   - Builds and publishes to PyPI (if configured)

## Development Workflow

```bash
# Clone and setup
git clone https://github.com/DASA-Design/PyDASA.git
cd PyDASA
git checkout dev

# Install in development mode
pip install -e ".[dev]"

# Make changes and test
pytest tests/

# Commit with conventional format
git commit -m "feat(module): add new feature"

# Push to dev branch
git push origin dev

# Create PR to main when ready for release
```

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes
- **MINOR** (0.X.0): New features (backward-compatible)
- **PATCH** (0.0.X): Bug fixes and improvements

Current version: **0.2.75**
