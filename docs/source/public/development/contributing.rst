Contributing to PyDASA
======================

Contributions are welcome! ⚠️ We use `Conventional Commits <https://www.conventionalcommits.org/>`_ ⚠️ for automatic versioning and changelog generation to maintain a clean and predictable release process.

Commit Message Format
---------------------

::

    <type>(<scope>): <subject>

**Types:**

- ``feat``: New feature (triggers MINOR version bump).
- ``fix``: Bug fix (triggers PATCH version bump).
- ``docs``: Documentation changes only.
- ``refactor``: Code refactoring without feature changes.
- ``test``: Adding or updating tests.
- ``perf``: Performance improvements.
- ``chore``: Other changes that don't modify src or test files.

**Breaking Changes:** Add ``BREAKING CHANGE:`` in commit footer to trigger MAJOR version bump.

Examples
--------

.. code-block:: bash

    # Feature (0.6.0 → 0.7.0)
    git commit -m "feat(workflows): add uncertainty propagation analysis"

    # Bug fix (0.6.0 → 0.6.1)
    git commit -m "fix(buckingham): resolve matrix singularity edge case"

    # Breaking change (0.6.0 → 1.0.0)
    git commit -m "feat(api)!: redesign Variable API

    BREAKING CHANGE: Variable.value renamed to Variable.magnitude"

Development Workflow
--------------------

.. code-block:: bash

    # Clone and setup
    git clone https://github.com/DASA-Design/PyDASA.git
    cd PyDASA

    # Install in development mode
    pip install -e ".[dev]"

    # Run tests
    pytest tests/

    # Commit with conventional format
    git commit -m "feat(module): add new feature"

    # Create PR for review

Release Process
---------------

1. Make changes with conventional commit messages.
2. Create PR and merge to ``main``.
3. GitHub Actions automatically:
    - Analyzes commit messages.
    - Bumps version (MAJOR.MINOR.PATCH).
    - Updates ``_version.py`` and ``pyproject.toml``.
    - Creates GitHub release with changelog.
    - Publishes to PyPI.
