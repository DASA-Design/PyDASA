# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.4] - 2026-01-07
### Fixed
- Downgrade Sphinx to 8.1.3 for myst-parser compatibility

### Changed
- Update Python version to 3.11 in Read the Docs configuration
- Configure Read the Docs for automated documentation builds
- Add PyPI publishing workflow with GitHub Actions

## [0.3.3] - 2026-01-07
### Added
- First public release published to PyPI
- Configure automated PyPI publishing via GitHub Actions
- Add Read the Docs integration with .readthedocs.yaml
- Include CHANGELOG.md in documentation via myst-parser

### Changed
- Update documentation structure with public folder integration
- Configure linkify-it-py for markdown link rendering in docs

## [0.3.2] - 2026-01-06
### Fixed
- Package now includes `core/cfg/*.json` configuration files in distribution
- `__version__` attribute is now properly exposed in the main `pydasa` module
- Resolved `FileNotFoundError` when importing pydasa after installation

### Added
- Added `[tool.setuptools.package-data]` configuration to include non-Python files
- Comprehensive test suite in `tests/test_package.py` to verify package initialization
- Tests for version accessibility, configuration file inclusion, and main exports

### Changed
- Configured semantic release with `major_on_zero = false` to keep versions in 0.x range
- Updated `__all__` list to include `__version__` for proper wildcard imports

## [0.3.1] - 2026-01-06
### Changed
- Version reset from 1.0.x to 0.3.1 to maintain pre-release versioning
- Configured semantic release to prevent automatic major version bumps

## [0.3.0] - Previous
### Added
- Initial stable release with core functionality

## [0.2.99] - Unreleased
### Added
- Distribution files: `pydasa-0.2.99-py3-none-any.whl`, `pydasa-0.2.99.tar.gz`

## [0.2.75] - Unreleased
### Added
- Distribution files: `pydasa-0.2.75-py3-none-any.whl`, `pydasa-0.2.75.tar.gz`

## [0.2.71] - Unreleased
### Added
- Distribution files: `pydasa-0.2.71-py3-none-any.whl`, `pydasa-0.2.71.tar.gz`

## [0.2.69] - Unreleased
### Added
- Distribution files: `pydasa-0.2.69-py3-none-any.whl`, `pydasa-0.2.69.tar.gz`

## [0.2.45] - Unreleased
### Added
- Distribution files: `pydasa-0.2.45-py3-none-any.whl`, `pydasa-0.2.45.tar.gz`

## [0.2.43] - Unreleased
### Added
- Distribution files: `pydasa-0.2.43-py3-none-any.whl`, `pydasa-0.2.43.tar.gz`

## [0.2.42] - Unreleased
### Added
- Distribution files: `pydasa-0.2.42-py3-none-any.whl`, `pydasa-0.2.42.tar.gz`

## [0.2.37] - Unreleased
### Added
- Distribution files: `pydasa-0.2.37-py3-none-any.whl`, `pydasa-0.2.37.tar.gz`

## [0.2.21] - Unreleased
### Added
- Distribution files: `pydasa-0.2.21-py3-none-any.whl`, `pydasa-0.2.21.tar.gz`

## [0.2.17] - Unreleased
### Added
- Distribution files: `pydasa-0.2.17-py3-none-any.whl`, `pydasa-0.2.17.tar.gz`

## [0.2.11] - Unreleased
### Added
- Distribution files: `pydasa-0.2.11-py3-none-any.whl`, `pydasa-0.2.11.tar.gz`

## [0.2.0] - Unreleased
### Added
- Distribution files: `pydasa-0.2.0-py3-none-any.whl`, `pydasa-0.2.0.tar.gz`

## [0.1.97] - Unreleased
### Added
- Distribution files: `pydasa-0.1.97-py3-none-any.whl`, `pydasa-0.1.97.tar.gz`

## [0.1.95] - Unreleased
### Added
- Distribution files: `pydasa-0.1.95-py3-none-any.whl`, `pydasa-0.1.95.tar.gz`

## [0.1.92] - Unreleased
### Added
- Distribution files: `pydasa-0.1.92-py3-none-any.whl`, `pydasa-0.1.92.tar.gz`

## [0.1.90] - Unreleased
### Added
- Distribution files: `pydasa-0.1.90-py3-none-any.whl`, `pydasa-0.1.90.tar.gz`

## [0.1.9] - Unreleased
### Added
- Distribution files: `pydasa-0.1.9-py3-none-any.whl`, `pydasa-0.1.9.tar.gz`

## [0.1.8] - Unreleased
### Added
- Distribution files: `pydasa-0.1.8-py3-none-any.whl`, `pydasa-0.1.8.tar.gz`

## [0.1.73] - Unreleased
### Added
- Distribution files: `pydasa-0.1.73-py3-none-any.whl`, `pydasa-0.1.73.tar.gz`

## [0.1.70] - Unreleased
### Added
- Distribution files: `pydasa-0.1.70-py3-none-any.whl`, `pydasa-0.1.70.tar.gz`

## [0.1.7] - Unreleased
### Added
- Distribution files: `pydasa-0.1.7-py3-none-any.whl`, `pydasa-0.1.7.tar.gz`

## [0.1.65] - Unreleased
### Added
- Distribution files: `pydasa-0.1.65-py3-none-any.whl`, `pydasa-0.1.65.tar.gz`

## [0.1.6] - Unreleased
### Added
- Distribution files: `pydasa-0.1.6-py3-none-any.whl`, `pydasa-0.1.6.tar.gz`

## [0.1.5] - Unreleased
### Added
- Distribution files: `pydasa-0.1.5-py3-none-any.whl`, `pydasa-0.1.5.tar.gz`

## [0.1.45] - Unreleased
### Added
- Distribution files: `pydasa-0.1.45-py3-none-any.whl`, `pydasa-0.1.45.tar.gz`

## [0.1.35] - Unreleased
### Added
- Distribution files: `pydasa-0.1.35-py3-none-any.whl`, `pydasa-0.1.35.tar.gz`

## [0.1.32] - Unreleased
### Added
- Distribution files: `pydasa-0.1.32-py3-none-any.whl`, `pydasa-0.1.32.tar.gz`

## [0.1.27] - Unreleased
### Added
- Distribution files: `pydasa-0.1.27-py3-none-any.whl`, `pydasa-0.1.27.tar.gz`

## [0.1.25] - Unreleased
### Added
- Distribution files: `pydasa-0.1.25-py3-none-any.whl`, `pydasa-0.1.25.tar.gz`

## [0.1.2] - Unreleased
### Added
- Distribution files: `pydasa-0.1.2-py3-none-any.whl`, `pydasa-0.1.2.tar.gz`

## [0.1.19] - Unreleased
### Added
- Distribution files: `pydasa-0.1.19-py3-none-any.whl`, `pydasa-0.1.19.tar.gz`

## [0.1.17] - Unreleased
### Added
- Distribution files: `pydasa-0.1.17-py3-none-any.whl`, `pydasa-0.1.17.tar.gz`

## [0.1.15] - Unreleased
### Added
- Distribution files: `pydasa-0.1.15-py3-none-any.whl`, `pydasa-0.1.15.tar.gz`

## [0.1.12] - Unreleased
### Added
- Distribution files: `pydasa-0.1.12-py3-none-any.whl`, `pydasa-0.1.12.tar.gz`

## [0.1.11] - Unreleased
### Added
- Distribution files: `pydasa-0.1.11-py3-none-any.whl`, `pydasa-0.1.11.tar.gz`

## [0.1.0] - Unreleased
### Added
- Distribution files: `pydasa-0.1.0-py3-none-any.whl`, `pydasa-0.1.0.tar.gz`

## [0.0.1] - Unreleased
### Added
- Distribution files: `pydasa-0.0.1-py3-none-any.whl`, `pydasa-0.0.1.tar.gz`
