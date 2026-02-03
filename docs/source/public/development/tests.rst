Running Tests
=============

**PyDASA** uses `pytest <https://pytest.org/>`_ as its testing framework to ensure code quality and reliability across all modules.

Test Structure
--------------

The test suite is organized to mirror the package structure:

.. code-block:: text

    tests/
    └── pydasa/
        ├── analysis/       # Analysis module tests
        ├── context/        # Unit system tests
        ├── core/           # Core functionality tests
        ├── data/           # Test data and fixtures
        ├── dimensional/    # Buckingham Pi theorem tests
        ├── elements/       # Variable and parameter tests
        ├── modules/        # Module-specific tests
        ├── serialization/  # Parser and serialization tests
        ├── structs/        # Data structure tests
        ├── validations/    # Validation system tests
        └── workflows/      # Workflow orchestration tests

Running All Tests
-----------------

To run the complete test suite:

.. code-block:: bash

    # From the project root directory
    pytest tests/

With verbose output:

.. code-block:: bash

    pytest tests/ -v

Running Specific Tests
----------------------

Run tests for a specific module:

.. code-block:: bash

    # Test specific module
    pytest tests/pydasa/workflows/

    # Test specific file
    pytest tests/pydasa/workflows/test_basic.py

    # Test specific test class
    pytest tests/pydasa/workflows/test_basic.py::TestWorkflowBase

    # Test specific test method
    pytest tests/pydasa/workflows/test_basic.py::TestWorkflowBase::test_initialization

Test Coverage
-------------

The project maintains high test coverage across core modules. Coverage reports are automatically generated and tracked via `Codecov <https://codecov.io/gh/DASA-Design/PyDASA>`_.

To generate a local coverage report:

.. code-block:: bash

    # Install coverage tools
    pip install pytest-cov

    # Run tests with coverage
    pytest tests/ --cov=pydasa --cov-report=html

    # View the report (opens in browser)
    # The report will be in htmlcov/index.html

Coverage Status
^^^^^^^^^^^^^^^

Current coverage status by module:

- ✅ **core/**: Foundation classes, configuration, I/O
- ✅ **dimensional/**: Buckingham Pi theorem, dimensional matrix solver
- ✅ **elements/**: Variable and parameter management
- ✅ **workflows/**: AnalysisEngine, MonteCarloSimulation, SensitivityAnalysis
- ✅ **validations/**: Decorator-based validation system
- ✅ **serialization/**: LaTeX and formula parsing
- ⚠️ **context/**: Unit conversion system (partial coverage - stub implementation)
- ⚠️ **structs/**: Data structures (partial test coverage)

Writing Tests
-------------

Test Conventions
^^^^^^^^^^^^^^^^

**PyDASA** tests follow these conventions:

1. **Test files**: Named ``test_*.py`` to be discovered by pytest
2. **Test classes**: Named ``TestClassName`` using unittest.TestCase
3. **Test methods**: Named ``test_method_name`` describing what is tested
4. **Assertions**: Use both unittest assertions and pytest assertions

Example Test Structure
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import unittest
    import pytest
    from pydasa.workflows.basic import WorkflowBase
    from pydasa.elements.parameter import Variable

    class TestWorkflowBase(unittest.TestCase):
        """Tests for WorkflowBase class."""

        def setUp(self):
            """Set up test fixtures."""
            self.workflow = WorkflowBase(_idx=0, _fwk="PHYSICAL")

        def test_initialization(self):
            """Test workflow initialization."""
            assert self.workflow._idx == 0
            assert self.workflow._fwk == "PHYSICAL"

        def test_variable_management(self):
            """Test adding and retrieving variables."""
            var = Variable(_sym="x", _dims="L")
            self.workflow.add_variable(var)
            assert "x" in self.workflow.variables

        def tearDown(self):
            """Clean up after tests."""
            pass

Test Data and Fixtures
^^^^^^^^^^^^^^^^^^^^^^

Test data is centralized in ``tests/pydasa/data/test_data.py`` for consistency:

.. code-block:: python

    from tests.pydasa.data.test_data import get_simulation_test_data

    def test_with_fixture():
        """Test using shared test data."""
        data = get_simulation_test_data()
        # Use data in your test
        assert data is not None

Continuous Integration
----------------------

Tests are automatically run on every commit via GitHub Actions:

- ✅ All tests must pass before merging
- ✅ Coverage reports are uploaded to Codecov
- ✅ Test results are visible in pull requests

Test Requirements
-----------------

Testing dependencies are specified in ``pyproject.toml``:

.. code-block:: toml

    [project.optional-dependencies]
    dev = [
        "pytest>=8.1.1",
        "pytest-cov>=4.0.0",  # For coverage reports
    ]

Install testing dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

Troubleshooting Tests
---------------------

Common Issues
^^^^^^^^^^^^^

**Import Errors**
    Install **PyDASA** in development mode: ``pip install -e .``

**Missing Dependencies**
    Install dev dependencies: ``pip install -e ".[dev]"``

**Test Discovery Issues**
    Ensure test files are named ``test_*.py`` and located in ``tests/`` directory

**Failed Tests After Changes**
    Run specific test to see detailed error: ``pytest path/to/test_file.py -v``

Debug Mode
^^^^^^^^^^

Run tests with detailed output and stop at first failure:

.. code-block:: bash

    # Stop on first failure
    pytest tests/ -x

    # Show local variables on failure
    pytest tests/ -l

    # Enter debugger on failure
    pytest tests/ --pdb

Best Practices
--------------

When writing tests for **PyDASA**:

1. **Test one thing at a time**: Each test method should verify a single behavior
2. **Use descriptive names**: Test names should clearly indicate what is being tested
3. **Include docstrings**: Explain what the test verifies
4. **Test edge cases**: Include tests for boundary conditions and error cases
5. **Keep tests independent**: Tests should not depend on execution order
6. **Use fixtures appropriately**: Share setup code via setUp/tearDown or pytest fixtures
7. **Mock external dependencies**: Isolate unit tests from external systems
8. **Maintain coverage**: Aim for high coverage on core functionality

Contributing Tests
------------------

When contributing to **PyDASA**:

1. **Add tests for new features**: All new code should include tests
2. **Update tests for changes**: Modify existing tests when changing functionality
3. **Run full test suite**: Execute ``pytest tests/`` before committing
4. **Check coverage**: Ensure new code maintains or improves coverage
5. **Follow conventions**: Match existing test structure and naming

See the :doc:`contributing` guide for more details on the development workflow.
