Sensitivity Analysis
====================

Overview
---------

The ``SensitivityAnalysis`` class provides high-level interface to sensitivity analyses on dimensionless coefficients in **PyDASA**, abstracting mathematical computations with symbolic and numerical modes. Identifies variables with significant impact on system behavior for parameter optimization and uncertainty quantification.

The ``SensitivityAnalysis`` simplifies sensitivity workflows through three core features:

1. **Manages Instances**: Handles multiple ``Sensitivity`` objects (one per coefficient) internally.
2. **Enables Modes**: Performs symbolic differentiation (``SYM``) or numerical sampling (``NUM``).
3. **Integrates Workflows**: Accepts ``AnalysisEngine`` outputs directly for dimensional consistency.

.. warning::
    **Precondition**: ``SensitivityAnalysis`` requires a completed dimensional analysis workflow. The analysis must provide:

    - Dimensional coefficients (from ``AnalysisEngine.solve()`` or ``AnalysisEngine.run_analysis()``)
    - Variable definitions with numerical bounds (``std_min``, ``std_max``, ``std_mean``)
    - Consistent schema across all components

    Use the exact model from :doc:`analysis` as input to ensure proper configuration.

Reynolds Number (:math:`Re`) Sensitivity Analysis Example
-----------------------------------------------------------

This example demonstrates sensitivity analysis for the Reynolds number using the dimensional analysis workflow as precondition:

.. code-block:: python

    from pydasa import Variable, Schema, AnalysisEngine
    from pydasa.dimensional.fundamental import Dimension
    from pydasa.workflows.influence import SensitivityAnalysis

    # ========================================================================
    # STEP 1: Run Dimensional Analysis (Precondition)
    # ========================================================================

    # Define custom framework (T, M, L only)
    custom_fdus = [
        Dimension(_idx=0, _sym="T", _unit="s", _name="Time"),
        Dimension(_idx=1, _sym="M", _unit="kg", _name="Mass"),
        Dimension(_idx=2, _sym="L", _unit="m", _name="Length")
    ]
    schema = Schema(_fwk="CUSTOM", _fdu_lt=custom_fdus)

    # Define variables with numerical bounds for sensitivity analysis
    variables = {
        "\\rho": Variable(_name="Density",
                        _sym="\\rho",
                        _cat="IN",
                        _dims="M*L^-3",
                        _units="kg/m³",
                        _std_mean=1000.0,
                        _std_min=990.0,
                        _std_max=1020.0,
                        relevant=True,
                        _schema=schema),
        "v": Variable(_name="Velocity",
                    _sym="v",
                    _cat="OUT",
                    _dims="L*T^-1",
                    _units="m/s",
                    _std_mean=5.0,
                    _std_min=1.0,
                    _std_max=6.0,
                    relevant=True,
                    _schema=schema),
        "D": Variable(_name="Diameter",
                    _sym="D",
                    _cat="IN",
                    _dims="L",
                    _units="m",
                    _std_mean=0.05,
                    _std_min=0.04,
                    _std_max=0.06,
                    relevant=True,
                    _schema=schema),
        "\\mu": Variable(_name="Viscosity",
                        _sym="\\mu",
                        _cat="IN",
                        _dims="M*L^-1*T^-1",
                        _units="Pa·s",
                        _std_mean=0.001002,
                        _std_min=0.0009,
                        _std_max=0.0011,
                        relevant=True,
                        _schema=schema),
        "g": Variable(_name="Gravity",
                    _sym="g",
                    _cat="CTRL",
                    _dims="L*T^-2",
                    _units="m/s²",
                    _std_mean=9.81,
                    _std_min=9.80,
                    _std_max=9.82,
                    relevant=False,
                    _schema=schema)  # Excluded from matrix
    }

    # Create and run dimensional analysis
    engine = AnalysisEngine(_name="Reynolds Number Analysis",
                            _fwk="CUSTOM",
                            _schema=schema,
                            _variables=variables)

    engine.create_matrix()
    coefficients = engine.solve()

    print(f"Dimensional analysis complete: {list(coefficients.keys())}")

    # ========================================================================
    # STEP 2: Perform Sensitivity Analysis
    # ========================================================================

    # Create sensitivity analysis workflow
    sensitivity = SensitivityAnalysis(_name="Reynolds Sensitivity",
                                    _fwk="CUSTOM",
                                    _cat="SYM",
                                    _schema=schema,
                                    _variables=variables,
                                    _coefficients=coefficients)

    # Run symbolic sensitivity analysis at mean values
    results = sensitivity.analyze_symbolic(val_type="mean")

    # Display results for Reynolds number coefficient
    re_key = "SEN_{\\Pi_{0}}"
    print("\n=== Symbolic Sensitivity Analysis Results ===")
    if re_key in results:
        print(f"\nSensitivity for {re_key}:")
        for var_sym, sensitivity_val in results[re_key].items():
            if var_sym in variables:
                var_name = variables[var_sym].name
            else:
                var_name = var_sym
            print(f"\t∂π/∂{var_sym} ({var_name}): {sensitivity_val:+.4e}")

    # ========================================================================
    # STEP 3: Numerical Sensitivity (Alternative)
    # ========================================================================

    # Switch to numerical analysis mode
    sensitivity.cat = "NUM"

    # Run FAST (Fourier Amplitude Sensitivity Test)
    numerical_results = sensitivity.analyze_numeric(n_samples=1000)

    print("\n=== Numerical Sensitivity Analysis Results (FAST) ===")
    for coeff_key, sens_data in numerical_results.items():
        print(f"\nSensitivity for {coeff_key}:")
        if "S1" in sens_data:
            # First-order sensitivity indices
            s1_indices = sens_data["S1"]
            for i, var_sym in enumerate(sensitivity.variables.keys()):
                if var_sym in variables:
                    var_name = variables[var_sym].name
                else:
                    var_name = var_sym
                
                # S1 is a list, access by index
                # S1 represent the contribution of each input variable to the output variance
                if i < len(s1_indices):
                    s1_val = s1_indices[i]
                    print(f"\tS1: [{var_sym}] ({var_name}): {s1_val:.4f}")

Displayed Capabilities
-----------------------

In the example we appreciate the following **PyDASA** capabilities:

1. **Requires Precondition**: Needs a completed dimensional analysis with solved coefficients.
2. **Creates Instances**: Generates internal ``Sensitivity`` objects (one per coefficient) automatically.
3. **Switches Modes**: Toggles ``cat="SYM"`` for symbolic differentiation or ``cat="NUM"`` for numerical sampling (FAST).
4. **Extracts Bounds**: Pulls ``std_mean``, ``std_min``, ``std_max`` from variables for evaluation.
5. **Quantifies Impacts**: Computes partial derivatives (symbolic) or variance indices (numerical) per variable.

Identifies critical parameters affecting dimensionless behavior for experimental design and model refinement.