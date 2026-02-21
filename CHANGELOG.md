# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.26][0.6.26] - 2026-02-21

### Fixed

- Updated license format in pyproject.toml for improved PyPI display

## [0.6.25][0.6.25] - 2026-02-21

### Fixed

- Updated license format in pyproject.toml and added GPLv3+ classifier for proper PyPI badge display

## [0.6.24][0.6.24] - 2026-02-21

### Fixed

- Updated setuptools requirement to >=61.0.0 for PEP 639 compliance

## [0.6.23][0.6.23] - 2026-02-21

### Fixed

- Corrected typo in setup.py comments and adjusted formatting for clarity

## [0.6.22][0.6.22] - 2026-02-20

### Fixed

- Updated setuptools requirement to >=61.0.0 for proper PEP 639 license expression support
- This fixes PyPI license badge displaying "invalid"

## [0.6.21][0.6.21] - 2026-02-20

### Fixed

- Removed license classifier from pyproject.toml to comply with PEP 639 (license expression supersedes classifiers)
- Removed duplicate license field from setup.py to avoid conflict with pyproject.toml
- Updated coverage badge format to new codecov.io URL structure

## [0.6.20][0.6.20] - 2026-02-20

### Fixed

- Removed duplicate license classifier from pyproject.toml and updated setup.py to reference it properly

## [0.6.19][0.6.19] - 2026-02-20

### Fixed

- Added missing license classifier to pyproject.toml for proper PyPI license badge display

## [0.6.18][0.6.18] - 2026-02-20

### Fixed

- ReadTheDocs dependency conflicts resolved through systematic version downgrades:
  - Sphinx 9.0.4 → 8.1.3 (myst-parser compatibility)
  - astroid 4.0.2 → 3.3.8 (sphinx-autoapi compatibility with Python 3.11)
  - pylint 4.0.4 → 3.3.4 (astroid 3.x compatibility)
  - isort 7.0.0 → 6.1.0 (pylint 3.3.4 compatibility)
  - sphinx-autodoc-typehints 3.6.0 → 2.4.4 (Sphinx 8.x compatibility)
- Removed duplicate git install line from requirements.txt

## [0.6.17][0.6.17] - 2026-02-20

### Fixed

- Downgraded isort version from 7.0.0 to 6.1.0 for compatibility with pylint 3.3.4

## [0.6.16][0.6.16] - 2026-02-20

### Fixed

- Downgraded astroid to 3.3.8 and pylint to 3.3.4 for compatibility with sphinx-autoapi

## [0.6.15][0.6.15] - 2026-02-20

### Fixed

- Downgraded Sphinx version from 9.0.4 to 8.1.3 for compatibility with myst-parser

## [0.6.14][0.6.14] - 2026-02-20

### Fixed

- Updated CHANGELOG, setup.py and requirements.txt for version consistency

## [0.6.13][0.6.13] - 2026-02-20

### Fixed

- PyPI package metadata: Corrected license format to display properly on PyPI (GPLv3+)
- README badges: Updated to clickable links with proper codecov badge URL
- Notebook tests: Cleared outputs from tutorial notebooks for CI compatibility

## [0.6.12][0.6.12] - 2026-02-20

### Fixed

- PyPI package metadata: Corrected license format to display properly on PyPI (GPLv3+)
- README badges: Updated to clickable links with proper codecov badge URL

## [0.6.11][0.6.11] - 2026-02-20

### Added

- Comprehensive test coverage improvements across core modules (error.py, memory.py, functions.py, symbolic.py, simulation.py, conceptual.py)
- Test coverage increased to 90-100\% across all enhanced modules

### Fixed

- Type safety improvements in test suite (`Optional[List[int]]` handling)
- Enhanced error handling for edge cases in comparison and validation functions

## [0.6.10][0.6.10] - 2026-02-10

### Fixed

- Fixed switcher.json JSON syntax error in stable documentation builds (removed invalid JSON comments from v0.6.9)

### Added

- Added CHANGELOG entries for versions 0.6.6-0.6.9 from git history

## [0.6.9][0.6.9] - 2026-02-10

### Fixed

- Updated switcher.json to remove deprecated version entry for v0.4.10
- Improved documentation and method naming for clarity
- Removed unused code

### Changed

- Continued development of custom dimensional framework tutorial

## [0.6.8][0.6.8] - 2026-02-09

### Added

- Queueing module for customization with enhanced Sphinx documentation configuration

### Changed

- Updated variable symbol for density in tutorial documentation from `\rho` to `\rho_{1}` to reflect updated notation

### Fixed

- Improved documentation clarity and consistency
- Enhanced LaTeX parsing logic

## [0.6.7][0.6.7] - 2026-02-06

### Fixed

- Refactored documentation and improved clarity across PyDASA modules

### Changed

- Enhanced Sphinx documentation configuration
- Enhanced documentation for dimensional analysis features in PyDASA
- Renamed `relevant_lt` to `relevance_lt` for consistency and clarity

## [0.6.6][0.6.6] - 2026-02-03

### Fixed

- Forced documentation update
- Updated CHANGELOG for version 0.6.5 with new features, changes, fixes, and removals

## [0.6.5][0.6.5] - 2026-02-03

### Added

- Comprehensive ReadTheDocs documentation structure with roadmap.rst and tests.rst
- Tutorial.rst for complete dimensional analysis workflow

### Changed

- Enhanced README and pyproject.toml descriptions
- Improved code structure and readability
- Updated notebook outputs for better readability

### Fixed

- Property validation in MonteCarlo simulation class
  - Fixed `summary` property to access `_count` directly without validation
  - Fixed `data` property getter to handle pre-allocated data with NaN arrays
- Calculate coefficients method in AnalysisEngine to filter setpoints per coefficient using `var_dims.keys()`

### Removed

- Obsolete testing.rst and empty documentation files
- Dictionary comprehensions replaced with explicit loops for clarity

## [0.6.4][0.6.4] - 2026-01-26

### Fixed

- Codecov action configuration
- Notebook test outputs

### Changed

- Refactored test structure for improved maintainability

## [0.6.3][0.6.3] - 2026-01-26

### Fixed

- Enhanced MonteCarlo variable handling and serialization
- Updated Codecov action to v5 with improved coverage report upload

### Removed

- Removed unused Jupyter notebook PyDASA-Yoly.ipynb from examples directory

### Changed

- Re-enabled notebook testing in CI workflow

## [0.6.2][0.6.2] - 2026-01-26

### Fixed

- Improved MonteCarlo initialization and handling of non-init fields

### Changed

- Temporarily commented out notebook testing step in CI workflow

## [0.6.1][0.6.1] - 2026-01-26

### Added

- Quickstart guide and Reynolds number analysis example documentation

### Fixed

- Enhanced testing workflow with coverage reporting
- Updated README badges

## [0.6.0][0.6.0] - 2026-01-25

### Changed

- Refactored `MonteCarloSimulation` to inherit from `WorkflowBase` (same pattern as `AnalysisEngine`)
- Updated utility methods: `reset()` preserves coefficients, `clear()` resets everything, `to_dict()`/`from_dict()` use parent serialization

### Removed

- ~100 lines of duplicate code: attributes (`_variables`, `_coefficients`, `_results`), properties, and `_validate_dict()` method

### Improved

- Consistent inheritance pattern across all workflow classes (14/14 tests passing)

## [0.5.3][0.5.3] - 2026-01-19

### Added

- `allow_nan` parameter to `validate_type` decorator for `np.nan` validation
- `data` property (getter/setter) for `MonteCarlo` input data matrix with automatic list-to-array conversion
- `median` property to `BoundsSpecs` and `StandardizedSpecs` classes
- `calculate_setpoint()` method to `Coefficient` class with comprehensive validation

### Changed

- `MonteCarlo` now inherits from both `Foundation` and `BoundsSpecs` (multiple inheritance)
- `BoundsSpecs` setters accept `np.nan` values via `allow_nan=True` parameter
- `MonteCarlo` type annotations changed to `Optional[float]` to match `BoundsSpecs`
- `statistics` and `summary` properties return `Dict[str, Union[float, int, None]]`
- `_reset_statistics()` and `clear()` call `BoundsSpecs.clear(self)` for inherited attributes
- `AnalysisEngine` symbol formatting and serialization improved in `to_dict()` and `from_dict()`

### Removed

- `_variance` attribute from `MonteCarlo` (standard deviation is sufficient)
- Redundant property getters: `mean`, `median`, `dev`, `min_value`, `max_value` (now inherited from `BoundsSpecs`)
- ~120 lines of duplicate code through inheritance refactoring

### Fixed

- Type checker errors for property overrides in `MonteCarlo`
- Test assertion in `test_buckingham.py` to match error message: "std_setpoint is not defined"
- Type mismatch between `BoundsSpecs` (`float | None`) and `MonteCarlo` declarations
- `np.isnan()` type errors with proper `Optional[float]` annotations

### Improved

- Validation decorator system with `np.nan` support for statistical computations
- Architecture: `None` for user config, `np.nan` for computed results
- Test coverage: 26 MonteCarlo tests, 53 Buckingham tests, 330+ numerical spec tests passing

## [0.5.2][0.5.2] - 2026-01-15

### Fixed

- Fixed type annotations in `Sensitivity` class for better type safety
  - Changed `_sym_func` from `Callable` to `sp.Expr` to properly represent SymPy expressions
  - Updated `_exe_func` to `Union[Callable, Dict[str, Callable]]` to handle both analysis modes
  - Fixed `_symbols` type from `Dict[str, Symbol]` to `Dict[sp.Symbol, sp.Symbol]`
  - Split `_variables` into `Dict[str, Variable]` for objects and `_var_names: List[str]` for names
  - Fixed property return types to match attribute types
- Resolved type checking errors: "subs is not a known attribute", "Dict is not callable"
- Fixed "var is possibly unbound" error in exception handlers with proper initialization
- Removed trailing whitespace in docstrings

### Added

- New `var_names` property getter in `Sensitivity` class
- Added `_validate_sympy_expr()` validation method for SymPy expression validation
- Added `_validate_analysis_ready()` method to check if analysis can be performed
- Comprehensive test suite for `Sensitivity` class (56 tests total, up from 43)
  - Tests for `exe_func` property setter with callable and dict types
  - Tests for `schema` property setter and validation
  - Tests for `variables` property setter and validation
  - Tests for `_validate_sympy_expr()` method
  - Tests for `_validate_analysis_ready()` method with various failure scenarios
  - Tests for `var_names` property getter

### Changed

- Improved error messages in `Sensitivity` class for better debugging
- Updated `analyze_symbolically()` with better type guards and current variable tracking
- Updated `analyze_numerically()` with proper type guards using local variables
- Enhanced `_parse_expression()` method with proper type narrowing
- Refactored property decorators for consistency with validation patterns
- Replaced lambda expressions with proper function definitions for PEP 8 compliance

### Internal

- Added type guards to prevent runtime errors with Union types
- Improved exception handling with proper context preservation (`raise ... from e`)
- Enhanced type safety for SymPy integration throughout the module
- Used local variables to satisfy type checker in complex assignment scenarios

## [0.5.1][0.5.1] - 2026-01-14

### Changed

- Updated terminology from 'framework' to 'schema' in Matrix and AnalysisEngine classes for consistency
- Refined dimensional analysis workflow capabilities

### Fixed

- Fixed error message for invalid schema assignment in TestAnalysisEngine

## [0.5.0][0.5.0] - 2026-01-13

### Added

- Enhanced dimensional expression parsing and validation in `parser.py`
- Functions to extract coefficients, powered coefficients, and numeric constants from expressions
- New method `derive_coefficient` in `AnalysisEngine` for deriving coefficients from existing ones
- Comprehensive validation for AnalysisEngine schema input types

### Changed

- Improved dimensional operation computations and validations
- Updated regex patterns in `patterns.py` for better matching of numeric constants and operations
- Refactored `AnalysisEngine` in `phenomena.py` to support custom schemas
- Restructured analysis workflows for better modularity
- Enhanced unit tests for the `Matrix` class and `AnalysisEngine`
- Updated error messages for clarity and consistency across validation checks

### Fixed

- Fixed test assertions to match updated property names
- Corrected framework validation logic in multiple classes

## [0.4.10][0.4.10] - 2026-01-12

### Added

- New validation decorators `validate_list_types` and `validate_dict_types` in `pydasa.validations.decorators` module
- Comprehensive test suite for new validation decorators (8 new test cases)
- Tests for `Coefficient.get_data()` method in test_buckingham.py

### Changed

- Refactored `Coefficient` class to use new `@validate_list_types` and `@validate_dict_types` decorators
- Simplified `Coefficient.clear()` method to use `super().clear()` from parent classes (Foundation, BoundsSpecs)
- Updated test assertions to match new decorator error messages

### Fixed

- Updated variable test data keys from metric to setpoint
- Adjusted decorator validation range syntax

### Improved

- Reduced code duplication by replacing custom validation methods with reusable decorators
- Enhanced type validation for parameterized generics (Dict[str, Variable], List[int])
- Cleaner and more maintainable validation architecture

## [0.4.9][0.4.9] - 2026-01-10

### Changed

- Updated author information and email in setup.py
- Refactored `Dimension` class to use `Frameworks` enum

### Fixed

- Cleaned up redundant code in `Variable` class
- Revised comments in memory.py and functions.py
- Removed deprecated workflows/__init__.py file

## [0.4.8][0.4.8] - 2026-01-08

### Added

- `metric` and `std_metric` properties to `NumericalSpecs` class
- Comprehensive tests for new metric properties

### Fixed

- Enhanced numerical specifications with metric property support
- Updated test data structure to support metric properties

## [0.4.7][0.4.7] - 2026-01-07

### Changed

- Update CHANGELOG for versions 0.4.4, 0.4.5, and 0.4.6 releases

## [0.4.6][0.4.6] - 2026-01-07

### Fixed

- Downgrade sphinx-autodoc-typehints to 2.5.0 for Sphinx 8 compatibility

## [0.4.5][0.4.5] - 2026-01-07

### Fixed

- Downgrade isort to 6.1.0 for pylint compatibility

## [0.4.4][0.4.4] - 2026-01-07

### Fixed

- Downgrade isort to 6.1.2 for pylint compatibility

### Changed

- Update CHANGELOG for version 0.4.3 release

## [0.4.3][0.4.3] - 2026-01-07

### Fixed

- Downgrade astroid and pylint for sphinx-autoapi compatibility

## [0.4.2][0.4.2] - 2026-01-07

### Changed

- Minor release updates

## [0.4.1][0.4.1] - 2026-01-07

### Fixed

- Update PyPI token secret name in release workflow

## [0.4.0][0.4.0] - 2026-01-07

### Added

- Read the Docs configuration file (.readthedocs.yaml)
- Automated documentation building and hosting on Read the Docs

### Changed

- Update CHANGELOG for version 0.3.4 release

## [0.3.4][0.3.4] - 2026-01-07

### Fixed

- Downgrade Sphinx to 8.1.3 for myst-parser compatibility

### Changed

- Update Python version to 3.11 in Read the Docs configuration
- Configure Read the Docs for automated documentation builds
- Add PyPI publishing workflow with GitHub Actions

## [0.3.3][0.3.3] - 2026-01-07

### Added

- First public release published to PyPI
- Configure automated PyPI publishing via GitHub Actions
- Add Read the Docs integration with .readthedocs.yaml
- Include CHANGELOG.md in documentation via myst-parser

### Changed

- Update documentation structure with public folder integration
- Configure linkify-it-py for markdown link rendering in docs

## [0.3.2][0.3.2] - 2026-01-06

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

## [0.3.1][0.3.1] - 2026-01-06

### Changed

- Version reset from 1.0.x to 0.3.1 to maintain pre-release versioning
- Configured semantic release to prevent automatic major version bumps

## [0.3.0][0.3.0] - Unreleased (2026-01-06)

### Added

- Initial stable release with core functionality

## [0.2.99] - Unreleased (2026-01-06)

### Added

- Distribution files: `pydasa-0.2.99-py3-none-any.whl`, `pydasa-0.2.99.tar.gz`

## [0.2.75] - Unreleased (2026-01-04)

### Added

- Distribution files: `pydasa-0.2.75-py3-none-any.whl`, `pydasa-0.2.75.tar.gz`

## [0.2.71] - Unreleased (2025-11-27)

### Added

- Distribution files: `pydasa-0.2.71-py3-none-any.whl`, `pydasa-0.2.71.tar.gz`

## [0.2.69] - Unreleased (2025-11-24)

### Added

- Distribution files: `pydasa-0.2.69-py3-none-any.whl`, `pydasa-0.2.69.tar.gz`

## [0.2.45] - Unreleased (2025-11-23)

### Added

- Distribution files: `pydasa-0.2.45-py3-none-any.whl`, `pydasa-0.2.45.tar.gz`

## [0.2.43] - Unreleased (2025-11-23)

### Added

- Distribution files: `pydasa-0.2.43-py3-none-any.whl`, `pydasa-0.2.43.tar.gz`

## [0.2.42] - Unreleased (2025-11-23)

### Added

- Distribution files: `pydasa-0.2.42-py3-none-any.whl`, `pydasa-0.2.42.tar.gz`

## [0.2.37] - Unreleased (2025-11-23)

### Added

- Distribution files: `pydasa-0.2.37-py3-none-any.whl`, `pydasa-0.2.37.tar.gz`

## [0.2.21] - Unreleased (2025-11-18)

### Added

- Distribution files: `pydasa-0.2.21-py3-none-any.whl`, `pydasa-0.2.21.tar.gz`

## [0.2.17] - Unreleased (2025-11-18)

### Added

- Distribution files: `pydasa-0.2.17-py3-none-any.whl`, `pydasa-0.2.17.tar.gz`

## [0.2.11] - Unreleased (2025-11-18)

### Added

- Distribution files: `pydasa-0.2.11-py3-none-any.whl`, `pydasa-0.2.11.tar.gz`

## [0.2.0] - Unreleased (2025-11-17)

### Added

- Distribution files: `pydasa-0.2.0-py3-none-any.whl`, `pydasa-0.2.0.tar.gz`

## [0.1.97] - Unreleased (2025-11-17)

### Added

- Distribution files: `pydasa-0.1.97-py3-none-any.whl`, `pydasa-0.1.97.tar.gz`

## [0.1.95] - Unreleased (2025-11-17)

### Added

- Distribution files: `pydasa-0.1.95-py3-none-any.whl`, `pydasa-0.1.95.tar.gz`

## [0.1.92] - Unreleased (2025-10-11)

### Added

- Distribution files: `pydasa-0.1.92-py3-none-any.whl`, `pydasa-0.1.92.tar.gz`

## [0.1.90] - Unreleased (2025-10-11)

### Added

- Distribution files: `pydasa-0.1.90-py3-none-any.whl`, `pydasa-0.1.90.tar.gz`

## [0.1.9] - Unreleased (2025-09-01)

### Added

- Distribution files: `pydasa-0.1.9-py3-none-any.whl`, `pydasa-0.1.9.tar.gz`

## [0.1.8] - Unreleased (2025-08-28)

### Added

- Distribution files: `pydasa-0.1.8-py3-none-any.whl`, `pydasa-0.1.8.tar.gz`

## [0.1.73] - Unreleased (2025-10-03)

### Added

- Distribution files: `pydasa-0.1.73-py3-none-any.whl`, `pydasa-0.1.73.tar.gz`

## [0.1.70] - Unreleased (2025-10-03)

### Added

- Distribution files: `pydasa-0.1.70-py3-none-any.whl`, `pydasa-0.1.70.tar.gz`

## [0.1.7] - Unreleased (2025-08-25)

### Added

- Distribution files: `pydasa-0.1.7-py3-none-any.whl`, `pydasa-0.1.7.tar.gz`

## [0.1.65] - Unreleased (2025-10-02)

### Added

- Distribution files: `pydasa-0.1.65-py3-none-any.whl`, `pydasa-0.1.65.tar.gz`

## [0.1.6] - Unreleased (2025-08-25)

### Added

- Distribution files: `pydasa-0.1.6-py3-none-any.whl`, `pydasa-0.1.6.tar.gz`

## [0.1.5] - Unreleased (2025-08-24)

### Added

- Distribution files: `pydasa-0.1.5-py3-none-any.whl`, `pydasa-0.1.5.tar.gz`

## [0.1.45] - Unreleased (2025-09-30)

### Added

- Distribution files: `pydasa-0.1.45-py3-none-any.whl`, `pydasa-0.1.45.tar.gz`

## [0.1.35] - Unreleased (2025-09-30)

### Added

- Distribution files: `pydasa-0.1.35-py3-none-any.whl`, `pydasa-0.1.35.tar.gz`

## [0.1.32] - Unreleased (2025-09-30)

### Added

- Distribution files: `pydasa-0.1.32-py3-none-any.whl`, `pydasa-0.1.32.tar.gz`

## [0.1.27] - Unreleased (2025-09-29)

### Added

- Distribution files: `pydasa-0.1.27-py3-none-any.whl`, `pydasa-0.1.27.tar.gz`

## [0.1.25] - Unreleased (2025-09-29)

### Added

- Distribution files: `pydasa-0.1.25-py3-none-any.whl`, `pydasa-0.1.25.tar.gz`

## [0.1.2] - Unreleased (2025-08-24)

### Added

- Distribution files: `pydasa-0.1.2-py3-none-any.whl`, `pydasa-0.1.2.tar.gz`

## [0.1.19] - Unreleased (2025-09-19)

### Added

- Distribution files: `pydasa-0.1.19-py3-none-any.whl`, `pydasa-0.1.19.tar.gz`

## [0.1.17] - Unreleased (2025-09-19)

### Added

- Distribution files: `pydasa-0.1.17-py3-none-any.whl`, `pydasa-0.1.17.tar.gz`

## [0.1.15] - Unreleased (2025-09-18)

### Added

- Distribution files: `pydasa-0.1.15-py3-none-any.whl`, `pydasa-0.1.15.tar.gz`

## [0.1.12] - Unreleased (2025-09-11)

### Added

- Distribution files: `pydasa-0.1.12-py3-none-any.whl`, `pydasa-0.1.12.tar.gz`

## [0.1.11] - Unreleased (2025-09-10)

### Added

- Distribution files: `pydasa-0.1.11-py3-none-any.whl`, `pydasa-0.1.11.tar.gz`

## [0.1.0] - Unreleased (2025-08-20)

### Added

- Distribution files: `pydasa-0.1.0-py3-none-any.whl`, `pydasa-0.1.0.tar.gz`

## [0.0.1] - Unreleased (2025-08-20)

### Added

- Distribution files: `pydasa-0.0.1-py3-none-any.whl`, `pydasa-0.0.1.tar.gz`

## [0.0.0] - Initial Development (2023-08-27 to 2025-08-20)

### Added

- Initial commit and repository setup.
- Fundamental dimensional unit (FDU) class implementation.
- Proof of concept for dimensional analysis framework.
- Core data structures for PyDASA methodology.
- Initial documentation structure (English and Spanish).
- Base classes for dimensional parameters and fundamental data structures.
- Early test structure and experimental implementations.

### Note

Experimentation phase before formal semantic versioning, establishing core dimensional analysis concepts and data structures.

[0.6.22]: https://github.com/DASA-Design/PyDASA/compare/v0.6.21...v0.6.22
[0.6.21]: https://github.com/DASA-Design/PyDASA/compare/v0.6.19...v0.6.21
[0.6.19]: https://github.com/DASA-Design/PyDASA/compare/v0.6.18...v0.6.19
[0.6.18]: https://github.com/DASA-Design/PyDASA/compare/v0.6.15...v0.6.18
[0.6.15]: https://github.com/DASA-Design/PyDASA/compare/v0.6.13...v0.6.15
[0.6.13]: https://github.com/DASA-Design/PyDASA/compare/v0.6.12...v0.6.13
[0.6.12]: https://github.com/DASA-Design/PyDASA/compare/v0.6.11...v0.6.12
[0.6.11]: https://github.com/DASA-Design/PyDASA/compare/v0.6.10...v0.6.11
[0.6.10]: https://github.com/DASA-Design/PyDASA/compare/v0.6.9...v0.6.10
[0.6.9]: https://github.com/DASA-Design/PyDASA/compare/v0.6.8...v0.6.9
[0.6.8]: https://github.com/DASA-Design/PyDASA/compare/v0.6.7...v0.6.8
[0.6.7]: https://github.com/DASA-Design/PyDASA/compare/v0.6.6...v0.6.7
[0.6.6]: https://github.com/DASA-Design/PyDASA/compare/v0.6.5...v0.6.6
[0.6.5]: https://github.com/DASA-Design/PyDASA/compare/v0.6.4...v0.6.5
[0.6.4]: https://github.com/DASA-Design/PyDASA/compare/v0.6.3...v0.6.4
[0.6.3]: https://github.com/DASA-Design/PyDASA/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/DASA-Design/PyDASA/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/DASA-Design/PyDASA/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/DASA-Design/PyDASA/compare/v0.5.3...v0.6.0
[0.5.3]: https://github.com/DASA-Design/PyDASA/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/DASA-Design/PyDASA/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/DASA-Design/PyDASA/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/DASA-Design/PyDASA/compare/v0.4.10...v0.5.0
[0.4.10]: https://github.com/DASA-Design/PyDASA/compare/v0.4.9...v0.4.10
[0.4.9]: https://github.com/DASA-Design/PyDASA/compare/v0.4.8...v0.4.9
[0.4.8]: https://github.com/DASA-Design/PyDASA/compare/v0.4.7...v0.4.8
[0.4.7]: https://github.com/DASA-Design/PyDASA/compare/v0.4.6...v0.4.7
[0.4.6]: https://github.com/DASA-Design/PyDASA/compare/v0.4.5...v0.4.6
[0.4.5]: https://github.com/DASA-Design/PyDASA/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/DASA-Design/PyDASA/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/DASA-Design/PyDASA/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/DASA-Design/PyDASA/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/DASA-Design/PyDASA/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/DASA-Design/PyDASA/compare/v0.3.4...v0.4.0
[0.3.4]: https://github.com/DASA-Design/PyDASA/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/DASA-Design/PyDASA/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/DASA-Design/PyDASA/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/DASA-Design/PyDASA/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/DASA-Design/PyDASA/releases/tag/v0.3.0
