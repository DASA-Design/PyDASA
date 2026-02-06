Tutorial
===============

This tutorial will guide you through a complete dimensional analysis workflow using PyDASA, from basic dimensional analysis to advanced sensitivity and Monte Carlo simulations.

We'll use the **Reynolds number and pipe flow** as a practical example to demonstrate all four major PyDASA workflows.

.. contents:: Table of Contents
    :local:
    :depth: 2

Introduction
------------

**What You'll Learn:**

1. **AnalysisEngine** - Perform dimensional analysis using Buckingham Pi theorem
2. **SensitivityAnalysis** - Understand which variables have the most influence
3. **MonteCarloSimulation** - Quantify uncertainty in your dimensionless coefficients
4. **Data Export & Visualization** - Use results with matplotlib, pandas, and other libraries

**Running Example: Reynolds Number & Pipe Flow**

We'll analyze pipe flow, which depends on:

- **ρ** (rho) - Fluid density [kg/m³]
- **v** - Flow velocity [m/s]
- **D** - Pipe diameter [m]
- **μ** (mu) - Dynamic viscosity [Pa·s]
- **ΔP** - Pressure drop [Pa]
- **L** - Pipe length [m]
- **ε** (epsilon) - Absolute roughness [m]

This will generate three key dimensionless numbers:

- **Reynolds Number (:math:`Re`)** - Flow regime predictor
- **Darcy Friction Factor (f)** - Pressure loss coefficient
- **Relative Roughness (ε/D)** - Surface roughness effect

----

Stage 1: Dimensional Analysis with AnalysisEngine
--------------------------------------------------

The ``AnalysisEngine`` workflow automatically derives dimensionless coefficients
from your physical variables using the Buckingham Pi theorem.

Step 1.1: Import PyDASA
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pydasa.workflows.phenomena import AnalysisEngine
    from pydasa.elements.parameter import Variable
    import numpy as np

Step 1.2: Define Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define variables as dictionaries with all required attributes. Each variable needs:

- ``_idx`` - Unique index
- ``_sym`` - Symbol (use LaTeX for Greek letters like ``"\\rho"``)
- ``_fwk`` - Framework (``"PHYSICAL"`` uses M, L, T dimensions)
- ``_cat`` - Category: ``"IN"`` (input), ``"OUT"`` (output), or ``"CTRL"`` (control)
- ``relevant`` - Must be ``True`` to include in analysis
- ``_dims`` - Dimensional representation (e.g., ``"M*L^-3"`` for density)
- ``_setpoint`` - Numerical value for calculations

.. code-block:: python

    variables_dict = {
        # Fluid density: ρ [M/L^3]
        "\\rho": {
            "_idx": 0,
            "_sym": "\\rho",
            "_alias": "rho",
            "_fwk": "PHYSICAL",
            "_cat": "IN",
            "_name": "Density",
            "description": "Fluid density (water at 20°C)",
            "relevant": True,
            "_dims": "M*L^-3",
            "_units": "kg/m³",
            "_setpoint": 1000.0,
            "_std_setpoint": 1000.0,
            "_std_min": 990.0,
            "_std_max": 1020.0,
        },
        
        # Velocity: v [L/T]
        "v": {
            "_idx": 1,
            "_sym": "v",
            "_fwk": "PHYSICAL",
            "_cat": "IN",
            "_name": "Velocity",
            "description": "Flow velocity",
            "relevant": True,
            "_dims": "L*T^-1",
            "_units": "m/s",
            "_setpoint": 5.0,
            "_std_setpoint": 5.0,
        },
        
        # Pipe diameter: D [L]
        "D": {
            "_idx": 2,
            "_sym": "D",
            "_fwk": "PHYSICAL",
            "_cat": "IN",
            "_name": "Pipe Diameter",
            "relevant": True,
            "_dims": "L",
            "_units": "m",
            "_setpoint": 0.05,
            "_std_setpoint": 0.05,
        },
        
        # Dynamic viscosity: μ [M/(L·T)]
        "\\mu": {
            "_idx": 3,
            "_sym": "\\mu",
            "_fwk": "PHYSICAL",
            "_cat": "OUT",  # Exactly ONE output variable required
            "_name": "Dynamic Viscosity",
            "relevant": True,
            "_dims": "M*L^-1*T^-1",
            "_units": "Pa·s",
            "_setpoint": 0.001002,
            "_std_setpoint": 0.001002,
        },
        
        # Pressure drop: ΔP [M/(L·T^2)]
        "P": {
            "_idx": 4,
            "_sym": "P",
            "_fwk": "PHYSICAL",
            "_cat": "CTRL",
            "_name": "Pressure Drop",
            "relevant": True,
            "_dims": "M*L^-1*T^-2",
            "_units": "Pa",
            "_setpoint": 5000.0,
            "_std_setpoint": 5000.0,
        },
        
        # Pipe length: L [L]
        "L": {
            "_idx": 5,
            "_sym": "L",
            "_fwk": "PHYSICAL",
            "_cat": "CTRL",
            "_name": "Pipe Length",
            "relevant": True,
            "_dims": "L",
            "_units": "m",
            "_setpoint": 10.0,
            "_std_setpoint": 10.0,
        },
        
        # Absolute roughness: ε [L]
        "\\varepsilon": {
            "_idx": 6,
            "_sym": "\\varepsilon",
            "_fwk": "PHYSICAL",
            "_cat": "CTRL",
            "_name": "Absolute Roughness",
            "relevant": True,
            "_dims": "L",
            "_units": "m",
            "_setpoint": 0.000025,
            "_std_setpoint": 0.000025,
        }
    }

    # Convert to Variable objects
    variables = {
        sym: Variable(**params) for sym, params in variables_dict.items()
    }

**Important Notes:**

- Exactly **ONE** variable must have ``"_cat": "OUT"``
- All relevant variables must have ``"relevant": True``
- Use ``"M"``, ``"L"``, ``"T"`` in ``_dims`` for PHYSICAL framework

Step 1.3: Create Analysis Engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create engine
    engine = AnalysisEngine(
        _idx=0,
        _fwk="PHYSICAL",
        _name="Pipe Flow Analysis",
        description="Dimensional analysis for Reynolds number and friction factor"
    )
    
    # Add variables
    engine.variables = variables

Step 1.4: Run Analysis
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Execute dimensional analysis
    results = engine.run_analysis()
    
    print(f"Number of dimensionless groups: {len(engine.coefficients)}")
    
    # Display results
    for name, coeff in engine.coefficients.items():
        print(f"{name}: {coeff.pi_expr}")

**Expected Output:**

.. code-block:: text

    Number of dimensionless groups: 4
    \Pi_{0}: \mu/(\rho*v*D)
    \Pi_{1}: L/D
    \Pi_{2}: P/(\rho*v**2)
    \Pi_{3}: \varepsilon/D

Step 1.5: Derive Meaningful Coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The original Pi groups may not be physically meaningful. Derive common coefficients:

.. code-block:: python

    # Get Pi coefficient keys
    pi_keys = list(engine.coefficients.keys())
    
    # Derive Reynolds Number: Re = 1/Π₀ = ρvD/μ
    Re_coeff = engine.derive_coefficient(
        expr=f"1/{pi_keys[0]}",
        symbol="Re",
        name="Reynolds Number",
        description="Re = ρvD/μ - Predicts flow regime",
        idx=-1
    )
    
    # Derive Darcy Friction Factor: f = 2(D/L)(ΔP/(ρv²)) = 2·(1/Π₁)·Π₂
    f_coeff = engine.derive_coefficient(
        expr=f"2*{pi_keys[1]}^-1 * {pi_keys[2]}",
        symbol="Pd",
        name="Pressure Drop Ratio",
        description="Related to Darcy friction factor",
        idx=-1
    )
    
    # Relative Roughness: ε/D = Π₃
    rough_coeff = engine.derive_coefficient(
        expr=f"{pi_keys[3]}",
        symbol="\\epsilon/D",
        name="Relative Roughness",
        description="ε/D - Surface roughness effect",
        idx=-1
    )
    
    # Calculate numerical values
    Re_val = Re_coeff.calculate_setpoint()
    f_val = f_coeff.calculate_setpoint()
    rough_val = rough_coeff.calculate_setpoint()
    
    print(f"\nReynolds Number: {Re_val:.2e}")
    print(f"Pressure Drop Ratio: {f_val:.6f}")
    print(f"Relative Roughness: {rough_val:.2e}")

**Expected Output:**

.. code-block:: text

    Reynolds Number: 2.49e+05
    Pressure Drop Ratio: 0.040000
    Relative Roughness: 5.00e-04

----

Stage 2: Sensitivity Analysis
------------------------------

The ``SensitivityAnalysis`` workflow helps you understand which variables
most influence your dimensionless coefficients.

Step 2.1: Import and Create Analyzer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pydasa.workflows.influence import SensitivityAnalysis
    
    # Create sensitivity analyzer
    sensitivity = SensitivityAnalysis(
        _idx=0,
        _fwk="PHYSICAL",
        _name="Reynolds Sensitivity",
        _cat="SYM"  # Symbolic analysis
    )
    
    # Configure with engine results
    sensitivity.variables = engine.variables
    sensitivity.coefficients = engine.coefficients

Step 2.2: Symbolic Sensitivity Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compute partial derivatives at mean values:

.. code-block:: python

    # Run symbolic analysis
    symbolic_results = sensitivity.analyze_symbolic(val_type="mean")
    
    # Display for Reynolds Number
    Re_sensitivity = symbolic_results["SEN_{Re}"]
    
    print("Reynolds Number Sensitivity:")
    for var, sens_val in Re_sensitivity.items():
        if isinstance(sens_val, (int, float)):
            print(f"  ∂Re/∂{var}: {sens_val:+.4e}")

**Expected Output:**

.. code-block:: text

    Reynolds Number Sensitivity:
      ∂Re/∂ρ: +2.4900e+02
      ∂Re/∂v: +4.9800e+04
      ∂Re/∂D: +4.9800e+03
      ∂Re/∂μ: -2.4851e+08

Step 2.3: Numerical Sensitivity (FAST Method)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use variance-based FAST method for confidence intervals:

.. code-block:: python

    import random
    np.random.seed(42)
    random.seed(42)
    
    # Run numerical analysis
    numerical_results = sensitivity.analyze_numeric(n_samples=1000)
    
    # Display FAST sensitivity indices
    Re_fast = numerical_results["SEN_{Re}"]
    
    print("\nFAST Sensitivity for Reynolds Number:")
    print(f"{'Variable':<15} {'First-Order (S1)':<20} {'Total-Order (ST)':<20}")
    print("-" * 55)
    
    var_names = Re_fast["names"]
    S1_vals = Re_fast["S1"]
    ST_vals = Re_fast["ST"]
    
    for i, var_name in enumerate(var_names):
        print(f"{var_name:<15} {S1_vals[i]:<20.6f} {ST_vals[i]:<20.6f}")

**Expected Output:**

.. code-block:: text

    FAST Sensitivity for Reynolds Number:
    Variable        First-Order (S1)     Total-Order (ST)
    -------------------------------------------------------
    \rho            0.333333             0.333333
    v               0.333333             0.333333
    D               0.333333             0.333333
    \mu             0.000000             0.000000

**Interpretation:**

- **S1** (First-Order): Direct effect of variable
- **ST** (Total-Order): Total effect including interactions
- ρ, v, D each contribute equally (~33%) to Reynolds number
- μ has zero variance (kept constant in this setup)

----

Stage 3: Monte Carlo Simulation
--------------------------------

The ``MonteCarloSimulation`` workflow quantifies uncertainty in your
dimensionless coefficients through probabilistic sampling.

Step 3.1: Configure Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define probability distributions for each variable:

.. code-block:: python

    from pydasa.workflows.practical import MonteCarloSimulation
    import random
    
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Configure variable distributions
    for var_sym, var in engine.variables.items():
        if var_sym == "v":
            # Vary velocity uniformly from 1.0 to 5.0 m/s
            a, b = 1.0, 5.0
            var._dist_type = "uniform"
            var._dist_params = {"a": a, "b": b}
            var._dist_func = lambda a=a, b=b: random.uniform(a, b)
            
        elif var_sym == "D":
            # Small variation in diameter
            a, b = 0.049, 0.051
            var._dist_type = "uniform"
            var._dist_params = {"a": a, "b": b}
            var._dist_func = lambda a=a, b=b: random.uniform(a, b)
            
        elif var_sym == "\\rho":
            # Density variation around water at 20°C
            a, b = 980.0, 1020.0
            var._dist_type = "uniform"
            var._dist_params = {"a": a, "b": b}
            var._dist_func = lambda a=a, b=b: random.uniform(a, b)
            
        else:
            # Keep other variables constant
            cst = var.setpoint
            var._dist_type = "constant"
            var._dist_params = {"value": cst}
            var._dist_func = lambda cst=cst: cst

Step 3.2: Run Monte Carlo Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create simulation handler
    mc_handler = MonteCarloSimulation(
        _idx=0,
        _fwk="PHYSICAL",
        _name="Reynolds Monte Carlo",
        _cat="DIST",
        _experiments=500,
        _variables=engine.variables,
        _coefficients=engine.coefficients
    )
    
    # Run simulation
    mc_handler.run_simulation(iters=500)
    
    print(f"Simulation complete: {mc_handler.experiments} experiments")

Step 3.3: Extract and Analyze Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Get simulation keys
    all_keys = list(mc_handler.simulations.keys())
    derived_keys = [k for k in all_keys if not k.startswith('\\Pi_')]
    
    # Extract results
    pi_data = {}
    for pi_key in derived_keys:
        pi_sim_obj = mc_handler.get_simulation(pi_key)
        pi_results = pi_sim_obj.extract_results()
        pi_data[pi_key] = pi_results[pi_key]
    
    # Get Reynolds number simulation data
    Re_sim = pi_data["Re"]
    
    # Display statistics
    print(f"\nReynolds Number Statistics:")
    print(f"  Mean: {np.mean(Re_sim):.2e}")
    print(f"  Std Dev: {np.std(Re_sim):.2e}")
    print(f"  Min: {np.min(Re_sim):.2e}")
    print(f"  Max: {np.max(Re_sim):.2e}")
    print(f"  Range: {np.max(Re_sim) - np.min(Re_sim):.2e}")

**Expected Output:**

.. code-block:: text

    Reynolds Number Statistics:
      Mean: 1.24e+05
      Std Dev: 7.17e+04
      Min: 1.96e+04
      Max: 2.55e+05
      Range: 2.35e+05

----

Stage 4: Data Export & Visualization
-------------------------------------

PyDASA results can be exported to dictionaries for use with matplotlib,
pandas, seaborn, and other data analysis libraries.

Step 4.1: Export to Dictionaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Export coefficient data
    coeff_export = {}
    for name, coeff in engine.coefficients.items():
        coeff_export[name] = {
            "expression": str(coeff.pi_expr),
            "value": coeff.calculate_setpoint(),
            "variables": list(coeff.var_dims.keys()),
            "exponents": list(coeff.var_dims.values())
        }
    
    # Export variable data
    var_export = {}
    for sym, var in engine.variables.items():
        var_export[sym] = {
            "name": var.name,
            "dims": var.dims,
            "units": var.units,
            "setpoint": var.setpoint
        }

Step 4.2: Use with Pandas
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import pandas as pd
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create DataFrame from simulation results
    sim_df = pd.DataFrame({
        "Re": Re_sim,
        "epsilon_D": pi_data["\\epsilon/D"],
        "Pd": pi_data["Pd"]
    })
    
    # Statistical summary
    print(sim_df.describe())
    
    # Export to CSV
    sim_df.to_csv("data/reynolds_simulation.csv", index=False)
    print("CSV exported to data/reynolds_simulation.csv")

Step 4.3: Visualization with Matplotlib
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Histogram of Reynolds Number:**

.. code-block:: python

    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(Re_sim, bins=50, color="steelblue", 
            alpha=0.7, edgecolor="black")
    
    # Add flow regime boundaries
    ax.axvline(2300, color="green", linestyle="--", 
               linewidth=2, label="Laminar limit")
    ax.axvline(4000, color="red", linestyle="--", 
               linewidth=2, label="Turbulent limit")
    
    # Labels and formatting
    ax.set_xlabel("Reynolds Number (Re)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title(f"Reynolds Number Distribution (n={len(Re_sim)})", 
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

**Sensitivity Bar Chart:**

.. code-block:: python

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get FAST results
    Re_fast = numerical_results["SEN_{Re}"]
    var_names = Re_fast["names"]
    S1_vals = Re_fast["S1"]
    
    # Create bar chart
    y_pos = np.arange(len(var_names))
    ax.barh(y_pos, S1_vals, color="steelblue", 
            alpha=0.8, edgecolor="black")
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"${name}$" for name in var_names])
    ax.set_xlabel("First-Order Sensitivity Index (S1)", 
                  fontsize=12, fontweight="bold")
    ax.set_title("Variable Influence on Reynolds Number", 
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.show()

**Scatter Plot Matrix:**

.. code-block:: python

    from pandas.plotting import scatter_matrix
    
    # Create scatter matrix
    scatter_matrix(sim_df, figsize=(12, 12), 
                   alpha=0.6, diagonal="kde")
    plt.suptitle("Monte Carlo Simulation Results", 
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

Step 4.4: Export for External Tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import json
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Export to JSON for web applications
    export_data = {
        "coefficients": coeff_export,
        "variables": var_export,
        "simulation": {
            "n_samples": len(Re_sim),
            "Re_mean": float(np.mean(Re_sim)),
            "Re_std": float(np.std(Re_sim)),
            "Re_data": Re_sim.tolist()
        }
    }
    
    with open("data/pydasa_results.json", "w") as f:
        json.dump(export_data, f, indent=2)
    
    print("Results exported to data/pydasa_results.json")

----

Tips and Tricks
---------------

This section provides essential guidelines and best practices to help you avoid common mistakes and get the most out of PyDASA's features.

**Variable Definition**

Proper variable configuration is critical for successful dimensional analysis. Each variable requires specific attributes to function correctly in the PyDASA workflow.

- Always set exactly **ONE** variable with ``"_cat": "OUT"``
- Include ``"relevant": True`` for all variables in analysis
- Provide ``_setpoint`` and ``_std_setpoint`` for calculations
- Provide ``_std_min`` and ``_std_max`` for sensitivity analysis bounds
- Use descriptive ``_name`` and ``description`` fields

**Dimensional Consistency**

PyDASA automatically validates dimensional consistency across your variables. Understanding how to properly specify dimensions is essential for accurate analysis.

.. code-block:: python

    # PyDASA validates dimensions automatically
    # Use FDUs like M (Mass), L (Length), T (Time) for the PHYSICAL framework
    
    # Good: Velocity [L/T]
    "_dims": "L*T^-1"
    
    # Good: Density [M/L^3]
    "_dims": "M*L^-3"
    
    # Good: Force [M·L/T^2]
    "_dims": "M*L*T^-2"

**Distribution Configuration**

When running Monte Carlo simulations, each variable needs a properly configured probability distribution. This determines how the variable varies during the simulation.

    .. code-block:: python

        # For Monte Carlo, always set:
        # - _dist_type: "uniform", "normal", or "constant"
        # - _dist_params: dict with distribution parameters
        # - _dist_func: lambda function for sampling
        
        # Example: Uniform distribution
        var._dist_type = "uniform"
        var._dist_params = {"a": 1.0, "b": 5.0}
        var._dist_func = lambda: random.uniform(1.0, 5.0)

Common Pitfalls
---------------

**Missing Output Variable**

    .. code-block:: python

        # ❌ WRONG: No output variable
        variables = {
            "v": {"_cat": "IN", ...},
            "D": {"_cat": "IN", ...}
        }
        
        # ✅ CORRECT: Exactly one output
        variables = {
            "v": {"_cat": "IN", ...},
            "D": {"_cat": "IN", ...},
            "\\mu": {"_cat": "OUT", ...}  # One output
        }

**Inconsistent Dimensions**

    .. code-block:: python

        # ❌ WRONG: Incorrect dimension for velocity
        "v": {"_dims": "M*L^-1", ...}  # This is viscosity!
        
        # ✅ CORRECT: Velocity is L/T
        "v": {"_dims": "L*T^-1", ...}

**Forgot to Set Distribution**

    .. code-block:: python

        # ❌ WRONG: Missing distribution for Monte Carlo
        mc_handler = MonteCarloSimulation(...)
        mc_handler.run_simulation()  # Will fail!
        
        # ✅ CORRECT: Configure distributions first
        for var in engine.variables.values():
            var._dist_type = "uniform"
            var._dist_params = {"a": min_val, "b": max_val}
            var._dist_func = lambda: random.uniform(min_val, max_val)
        
        mc_handler.run_simulation()

1. **Ignoring Relevant Flag**

    .. code-block:: python

        # ❌ WRONG: Variable defined but not included
        "v": {"relevant": False, ...}  # Will be ignored!
        
        # ✅ CORRECT: Include in analysis
        "v": {"relevant": True, ...}

        # Good
        reynolds_number = pi_groups[0]
        
        # Better
        Re = pi_groups[0]
        Re.name = "Reynolds Number"
        Re.symbol = "Re"

Summary
-------

**Workflow Overview:**

.. list-table::
   :header-rows: 1
   :widths: 5 30 65

   * - Stage
     - Workflow
     - Key Actions
   * - 1
     - **AnalysisEngine**
     - Define variables → Create engine → Run analysis → Derive coefficients
   * - 2
     - **SensitivityAnalysis**
     - Configure analyzer → Run symbolic analysis → Run FAST method → Interpret indices
   * - 3
     - **MonteCarloSimulation**
     - Set distributions → Run simulation → Extract results → Compute statistics
   * - 4
     - **Export & Visualize**
     - Export to dict/DataFrame → Create plots → Save results → Generate reports

----

Next Steps
----------

Now that you've completed the tutorial, explore:

.. seealso::

    **Full Working Examples:**
    
    **Notebooks:**
    
    - :download:`Reynolds-Moody Diagram Analysis <../../_static/examples/notebooks/PyDASA-Reynolds.ipynb>`
    - :download:`Online Tutorial Notebook <../../_static/examples/notebooks/PyDASA-Online-Tutorial.ipynb>`
    - :download:`Yoly Example <../../_static/examples/notebooks/PyDASA-Yoly.ipynb>`
    
    **Code:**
    
    - :download:`reynolds_simple.py <../../_static/examples/code/reynolds_simple.py>`

**Documentation and User Guides:**

To understand **PyDASA** capabilities and internal theckincal details check:

- :doc:`../features/index` - Available features and workflows.
- :doc:`../examples/index` - More practical examples using PyDASA.
- :doc:`../../autoapi/index` - Complete API reference.

**Questions or Issues?**

To solve any doubt or bug, visit the `GitHub Issues Page <https://github.com/DASA-Design/PyDASA/issues>`_.
