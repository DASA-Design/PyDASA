Dimensionless Coefficients
==========================

Overview
--------

Dimensionless coefficients (:math:`\Pi`-groups) represent the fundamental output of dimensional analysis in **PyDASA**, transforming variable relationships into scale-invariant numerical expressions. The ``Coefficient`` class implements these dimensionless numbers following the Buckingham Pi-Theorem, providing the mathematical foundation for comparing systems across different scales, platforms, and operational contexts without dimensional constraints.

The ``Coefficient`` class inherits from ``BoundsSpecs`` (from ``numerical.py``), combining dimensionless mathematical properties with numerical ranges for computational analysis. This composite architecture bridges dimensional theory and practical evaluation, supporting sensitivity analysis through coefficient value exploration across different ranges, and Monte Carlo simulation through stochastic sampling of coefficient distributions derived from variable behaviour.

Dimensional Analysis Result
-----------------------------

Dimensionless coefficients result from solving the dimensional model (diagonalizing the dimensional matrix), producing :math:`\Pi`-groups with the the Buckingham Pi-Theorem. These coefficients serve three essential functions. One, represent Scale-Invariants by eliminating dimensional units through variable combinations with specific exponents. Two, reduce System Complexity by transforming n variables with m dimensions into exactly (n - m) dimensionless coefficients. And three, indicate Cross-Platform Equivalences by showing that identical coefficient values reflect equivalent system behavior regardless of scale or implementation.

Composite Architecture
----------------------

The ``Coefficient`` class combines three architectural layers to support dimensional analysis workflows:

1. **Foundation Layer** (from ``Foundation`` base class) Provides identity and classification attributes:
    - **Name** (``_name``): Identifies coefficient purpose (e.g., "Performance Ratio", "Mixing Coefficient").
    - **Symbol** (``_sym``): Represents coefficient in mathematical expressions (e.g., "π₁", "π₂").
    - **Index** (``_idx``): Determines coefficient precedence in analysis ordering.
    - **Framework** (``_fwk``): Links to dimensional framework (*PHYSICAL*, *COMPUTATION*, *SOFTWARE*, *CUSTOM*).
    - **Category** (``_cat``): Classifies as COMPUTED (from matrix solution) or DERIVED (from other coefficients).

2. **Numerical Layer** (from ``BoundsSpecs`` in ``numerical.py``) inherits standardized numerical attributes for computational analysis:
    - **Setpoint** (``_setpoint``): Specifies reference value for nominal operating conditions.
    - **Bounds** (``_min``, ``_max``): Define feasible coefficient range from variable combinations.
    - **Statistics** (``_mean``, ``_median``, ``_dev``): Store central tendency and dispersion measures.
    - **Step** (``_step``): Controls discretization granularity for sensitivity sweeps (default: 1e-3).
    
These attributes operate in dimensionless space, with values derived from standardized variable bounds through coefficient formulas.

3. **Coefficient-Specific Layer** Adds dimensional analysis properties:
    - **Variables** (``_variables``): Stores ``Variable`` objects participating in coefficient construction.
    - **Pi Expression** (``_pi_expr``): Contains symbolic formula (e.g., "R*g/v**2" for π₁ = R·g/v²).
    - **Variable Dimensions** (``var_dims``): Maps variable symbols to their exponents in coefficient formula.
    - **Dimensional Column** (``_dim_col``): Provides vector representation from RREF matrix solution.
    - **Pivot List** (``_pivot_lt``): Records pivot column indices identifying core/residual matrix structure.
    - **Data Array** (``_data``): Holds computed coefficient values from simulations or measurements.

This layered architecture connects dimensional theory (coefficient construction) with numerical computation (range-based analysis), forming the bridge between symbolic dimensional analysis and quantitative system evaluation.

Analytical Workflows
--------------------

The ``Coefficient`` class supports two primary computational workflows by inheriting numerical range capabilities from ``BoundsSpecs``:

**Sensitivity Analysis**: Variable bounds propagate through coefficient formulas to compute feasible ranges, which are discretized using the ``_step`` attribute for systematic parameter space exploration. This enables identifying which coefficients most strongly influence system behavior.

**Monte Carlo Simulation**: Variable distributions generate random samples that evaluate through coefficient formulas, producing coefficient value distributions. The ``_data`` array stores samples while ``_mean``, ``_median``, and ``_dev`` attributes quantify uncertainty propagation from variables to dimensionless coefficients.

Both workflows transform coefficients from symbolic expressions into computational objects with ranges, distributions, and statistical properties for quantitative system evaluation.

Practical Example
-----------------

Consider analyzing projectile motion where the ``Matrix`` class derives two coefficients:

.. code-block:: python

    import numpy as np
    from pydasa import Variable, Schema, Matrix

    # Schema and variables from Matrix example
    schema = Schema(_fwk="PHYSICAL")
    variables = {
        "R": Variable(_name="Range",
                        _sym="R", _cat="OUT",
                        _dims="L",
                        relevant=True,
                        _std_min=10, _std_max=100,
                        _units="m", _schema=schema),
        "v": Variable(_name="Velocity",
                        _sym="v", _cat="IN",
                        _dims="L*T^-1",
                        relevant=True,
                        _std_min=5, _std_max=50,
                        _units="m/s", _schema=schema),
        "g": Variable(_name="Gravity",
                        _sym="g", _cat="CTRL",
                        _dims="L*T^-2",
                        relevant=True,
                        _std_setpoint=9.81, _units="m/s^2",
                        _schema=schema)
    }

    # Generate dimensional matrix and coefficients
    model = Matrix(_name="Projectile",
                    _schema=schema,
                    _variables=variables)
    model.create_matrix()
    model.solve_matrix()  # Produces π₁ = R·g/v², π₂ = θ (dimensionless)

    # Access coefficient with inherited numerical properties
    pi_0 = model.coefficients["\Pi_{0}"]             # Coefficient object for pi_0
    print(f"Formula: {pi_0.pi_expr}")                # "R*g/v**2"
    print(f"Variables: {pi_0.variables.keys()}")     # dict_keys(['R', 'g', 'v'])
    print(f"Exponents: {pi_0.var_dims}")             # {'R': 1, 'g': 1, 'v': -2}
        
    # can setup bounds and discretization to compute data
    pi_0.min = variables["R"].std_min * variables["g"].std_setpoint / (variables["v"].std_max ** 2)  # (10*9.81)/(50**2) = 0.039
    pi_0.max = variables["R"].std_max * variables["g"].std_setpoint / (variables["v"].std_min ** 2)  # (100*9.81)/(5**2) = 39.24
    pi_0.step = 0.1  # Custom step for analysis
    pi_0.data = np.arange(pi_0.min, pi_0.max, pi_0.step)  # Grid data for analysis

    pi_0._data = np.array(data)
    pi_0._mean = np.mean(pi_0._data)    # Average dimensionless coefficient
    pi_0._dev = np.std(pi_0._data)      # Coefficient uncertainty
    print(f"Monte Carlo mean: {pi_0.mean}, std: {pi_0.dev}")

    # or generate data with Monte Carlo sampling using variable bounds
    data = []
    for _ in range(100):
        R_sample = np.random.uniform(variables["R"].std_min, variables["R"].std_max)
        v_sample = np.random.uniform(variables["v"].std_min, variables["v"].std_max)
        
        # Compute pi_0 for each sample
        point = pi_0.calculate_setpoint(dict(R=R_sample,
                                            v=v_sample,
                                            g=variables["g"].std_setpoint))
        # Compute pi_0 for each sample
        data.append(point)

    pi_0._data = np.array(data)
    pi_0._mean = np.mean(pi_0._data)    # Average dimensionless coefficient
    pi_0._dev = np.std(pi_0._data)      # Coefficient uncertainty
    print(f"Monte Carlo mean: {pi_0.mean}, std: {pi_0.dev}")

This example demonstrates how coefficients bridge dimensional theory and numerical computation: the ``Matrix`` class derives symbolic formulas (``_pi_expr``, ``var_dims``), while inherited ``BoundsSpecs`` attributes (``_min``, ``_max``, ``_step``, ``_mean``, ``_dev``, ``_data``) provide the numerical infrastructure for sensitivity analysis and Monte Carlo simulation without dimensional unit complications.