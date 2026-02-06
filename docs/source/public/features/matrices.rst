Dimensional Matrices & Models
==============================

Overview
--------

The ``Matrix`` class orchestrates dimensional analysis in **PyDASA** by implementing the Buckingham Pi-Theorem following Görtler's theoretical framework. This class transforms properly configured ``Variable`` collections and their ``Schema`` into dimensionless :math:`\Pi`-groups, represented by the ``Coefficient`` class.

The ``Matrix`` class requires two prerequisites: a well-configured ``Schema`` defining the dimensional framework (*FDUs*) and a ``relevance_lt`` (relevance list) containing variables marked for analysis. Without these, the ``Schema`` class cannot establish dimensional homogeneity rules and the operate over the relevance list defining system parameters.

Prerequisites for Analysis
--------------------------

The ``Matrix`` class operates on two essential inputs:

1. **Schema Configuration** is a properly initialized ``Schema`` instance defines the dimensional framework:
    - **FDU Collection** (``_fdu_lt``): Stores Fundamental Dimensional Units (e.g., ``L``, ``M``, ``T``, and so forth for *PHYSICAL*).
    - **FDU Mapping** (``_fdu_map``): Links FDU symbols to ``Dimension`` objects with precedence indices.
    - **Validation Patterns**: Provides regular expressions for dimensional expression parsing and validation.
        
The ``Schema`` ensures dimensional homogeneity across all variables, validating that dimensional expressions conform to the framework's rules.

2. **Relevance List** is the curated collection of ``Variable`` instances marked with ``relevant=True``:
    - **Dimensional Properties**: Requires each variable to have ``_dims`` (dimensional formula) and ``_dim_col`` (dimensional column vector).
    - **Category Assignment**: Classifies variables as *INPUT* (``IN``), *OUTPUT* (``OUT``), or *CONTROL* (``CTRL``).
    - **Completeness**: Ensures all variables affecting the system behavior are included.
        
The relevance list (``_relevant_lt``) determines matrix dimensions and consequently the number of computed :math:`\Pi`-coefficients (n_relevant - n_fdus).

Core Capabilities
-----------------

The ``Matrix`` class provides three primary operations for dimensional analysis:

1. **Matrix Construction** builds the dimensional matrix by extracting dimensional columns (``_dim_col``) from relevant variables and organizing them by precedence (*INPUT*, *OUTPUT*, *CONTROL*). This produces an (n_fdus × n_relevant) matrix where rows represent FDUs and columns represent variables. Variables with ``_cat="IN"`` constitute the core dimensional matrix, while ``_cat="OUT"`` and ``_cat="CTRL"`` form the residual matrix.
2. **Coefficient Derivation** solves the dimensional matrix using *Row-Reduced Echelon Form* (*RREF*) via *SymPy's* symbolic computation, identifying pivot columns as the "core" and free columns as the "residual". From this solution, the class derives dimensionless :math:`\Pi`-coefficients following Buckingham's theorem, each representing a dimensionally homogeneous variable combination.
3. **Validation & Verification** verifies :math:`\Pi`-groups sum to zero across all *FDU* dimensions, validates variable participation, and examines *INPUT*, *OUTPUT*, and *CONTROL* relationships. This ensures coefficients satisfy dimensional homogeneity requirements and represent system physics correctly.

Analytical Workflow
-------------------

The ``Matrix`` class supports a structured workflow for dimensional analysis:

1. **Initialization** accepts a configured ``Schema`` and populated ``_variables`` dictionary, automatically filtering relevant variables into ``_relevant_lt`` based on ``Variable.relevant`` flags, and extracting active FDUs (``working_fdus``) from variable dimensional expressions.
2. **Matrix Assembly** calls ``setup_dimensional_matrix()`` to construct ``_dim_mtx`` from variable dimensional columns, creates ``_sym_mtx`` (SymPy matrix) for symbolic operations, and validates matrix rank against FDU count to ensure solvability.
3. **Coefficient Generation** invokes ``calculate_coefficients()`` to compute RREF matrix (``_rref_mtx``), identify pivot columns (``_pivot_cols``), and derive :math:`\Pi`-coefficients stored in ``_coefficients`` dictionary with symbols like ``\Pi_{1}``, ``\Pi_{2}``, etc.
4. **Result Inspection** accesses derived coefficients through ``coefficients`` property, examines individual coefficient compositions via ``Coefficient`` objects, and validates results using ``check_coefficient_consistency()`` method.

This workflow ensures systematic application of dimensional analysis principles while maintaining traceability throughout the process.

Practical Example
-----------------

Consider analyzing projectile motion with PHYSICAL framework:

.. code-block:: python

    from pydasa import Variable, Schema, Matrix

    # Configure Schema
    schema = Schema(_fwk="PHYSICAL")

    # Simple example: Free fall
    # Define variables (only dimensional quantities)
    variables = {
        "h": Variable(_name="Height", _sym="h", _cat="OUT", 
                        _dims="L", _units="m", relevant=True, _schema=schema),
        "v": Variable(_name="Velocity", _sym="v", _cat="IN",
                        _dims="L*T^-1", _units="m/s", relevant=True, _schema=schema),
        "g": Variable(_name="Gravity", _sym="g", _cat="CTRL",
                        _dims="L*T^-2", _units="m/s^2", relevant=True, _schema=schema)
    }

    # Create and solve dimensional model
    model = Matrix(_name="Free Fall", _schema=schema, _variables=variables)
    model.create_matrix()
    model.solve_matrix()

    # Display results
    for sym, coef in model.coefficients.items():
        print(f"{sym}: {coef.pi_expr}")

This example demonstrates how a properly configured Schema and relevant variable list produce physically meaningful dimensionless coefficients representing projectile motion relationships.