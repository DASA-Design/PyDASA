Monte Carlo Simulation
======================

Overview
---------

The ``MonteCarloSimulation`` class provides high-level interface to uncertainty quantification for dimensionless coefficients in **PyDASA**, abstracting stochastic computations with distribution/data modes. Generates coefficient distributions under parameter variations for robustness assessment.

The ``MonteCarloSimulation`` simplifies stochastic workflows through three core features:

1. **Generates Simulations**: Creates ``MonteCarlo`` objects (one per coefficient) automatically.
2. **Samples Distributions**: Draws values from uniform, normal, or custom distributions.
3. **Accepts Inputs**: Takes ``AnalysisEngine`` outputs for dimensional consistency.

.. warning::
    **Precondition**: ``MonteCarloSimulation`` requires a completed dimensional analysis workflow. The analysis must provide:

    - Dimensional coefficients (from ``AnalysisEngine.solve()`` or ``AnalysisEngine.run_analysis()``)
    - Variable definitions with distribution specifications (``_dist_type``, ``_dist_params``, ``_dist_func``)
    - Consistent schema across all components

    Use the exact model from :doc:`analysis` as input to ensure proper configuration.

Reynolds Number (:math:`Re`) Monte Carlo Example
--------------------------------------------------

This example demonstrates Monte Carlo simulation for the Reynolds number using the dimensional analysis workflow as precondition:

.. code-block:: python

    from pydasa import Variable, Schema, AnalysisEngine
    from pydasa.dimensional.fundamental import Dimension
    from pydasa.workflows.practical import MonteCarloSimulation
    import random
    import numpy as np

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

    # Define variables with distribution specifications
    variables = {
        "\\rho": Variable(_name="Density",
                            _sym="\\rho",
                            _cat="IN",
                            _dims="M*L^-3",
                            _units="kg/m³",
                            _dist_type="uniform",
                            _dist_params={"a": 990.0, "b": 1020.0},
                            _dist_func=lambda: random.uniform(990.0, 1020.0),
                            relevant=True,
                            _schema=schema),
        "v": Variable(_name="Velocity",
                        _sym="v",
                        _cat="OUT",
                        _dims="L*T^-1",
                        _units="m/s",
                        _dist_type="uniform",
                        _dist_params={"a": 1.0, "b": 6.0},
                        _dist_func=lambda: random.uniform(1.0, 6.0),
                        relevant=True,
                        _schema=schema),
        "D": Variable(_name="Diameter",
                        _sym="D",
                        _cat="IN",
                        _dims="L",
                        _units="m",
                        _dist_type="uniform",
                        _dist_params={"a": 0.04, "b": 0.06},
                        _dist_func=lambda: random.uniform(0.04, 0.06),
                        relevant=True,
                        _schema=schema),
        "\\mu": Variable(_name="Viscosity",
                            _sym="\\mu",
                            _cat="IN",
                            _dims="M*L^-1*T^-1",
                            _units="Pa·s",
                            _dist_type="constant",
                            _dist_params={"value": 0.001002},
                            _dist_func=lambda: 0.001002,
                            relevant=True,
                            _schema=schema),
        "g": Variable(_name="Gravity",
                        _sym="g",
                        _cat="CTRL",
                        _dims="L*T^-2",
                        _units="m/s²",
                        _dist_type="constant",
                        _dist_params={"value": 9.81},
                        _dist_func=lambda: 9.81,
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
    # STEP 2: Perform Monte Carlo Simulation
    # ========================================================================

    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Create Monte Carlo simulation workflow
    mc_simulation = MonteCarloSimulation(_name="Reynolds Monte Carlo",
                                            _fwk="CUSTOM",
                                            _cat="DIST",
                                            _experiments=1000,
                                            _variables=variables,
                                            _coefficients=coefficients)

    # Run simulation
    mc_simulation.run_simulation(iters=1000)

    print(f"✓ Monte Carlo simulation complete")
    print(f"  Experiments: {mc_simulation.experiments}")
    print(f"  Simulations: {list(mc_simulation.simulations.keys())}")

    # ========================================================================
    # STEP 3: Extract Results and Statistics
    # ========================================================================

    # Get Reynolds number simulation results
    re_key = "\\Pi_{0}"
    re_simulation = mc_simulation.get_simulation(re_key)

    print(f"\nReynolds Number Statistics:")
    print(f"\tMean: {re_simulation.mean:.2e}")
    print(f"\tMedian: {re_simulation.median:.2e}")
    print(f"\tStd Dev: {re_simulation.dev:.2e}")
    print(f"\tMin: {re_simulation.min:.2e}")
    print(f"\tMax: {re_simulation.max:.2e}")

    # Extract result arrays for further analysis
    results_dict = mc_simulation.results
    re_results = results_dict[re_key]["results"]

    print(f"\nResult array shape: {re_results.shape}")
    print(f"\tFirst 5 results: {re_results[:5]}...")


Displayed Capabilities
-----------------------

In the example we appreciate the following **PyDASA** capabilities:

1. **Requires Precondition**: Demands completed dimensional analysis with solved coefficients.
2. **Generates Objects**: Creates ``MonteCarlo`` instances per coefficient automatically.
3. **Configures Sampling**: Sets distributions (uniform, constant) from variable specifications.
4. **Runs Iterations**: Executes Monte Carlo experiments with random sampling.
5. **Calculates Statistics**: Computes mean, median, std dev, min, max per coefficient.

Quantifies uncertainty propagation through dimensionless coefficients for robustness assessment.