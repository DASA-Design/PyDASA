Customization
=============

This tutorial demonstrates how to create and use custom dimensional frameworks in PyDASA for specialized applications. You'll learn how to define your own fundamental dimensions and apply dimensional analysis to non-traditional engineering domains.

We'll analyze a **queueing system (M/M/c/K)** using a simplified custom framework with only **Time** and **Structure** dimensions, making it analogous to classic dimensionless analysis like the Reynolds number.

.. contents:: Table of Contents
    :local:
    :depth: 2

Introduction
------------

**What You'll Learn:**

1. **Custom Schema Definition** - Create your own dimensional framework with custom FDUs
2. **Variable Definition with Custom Dimensions** - Define variables using your custom framework
3. **Dimensional Analysis** - Apply Buckingham Pi theorem to derive dimensionless groups
4. **Data Generation** - Generate systematic data from a queueing model
5. **Monte Carlo Simulation** - Run probabilistic analysis with grid-based data
6. **Advanced Visualization** - Create comprehensive 3D and 2D plots

**Running Example: M/M/c/K Queue Model**

We'll analyze a queueing system with:

- **λ** (lambda) - Arrival rate [requests/s]
- **K** - Queue capacity [requests]
- **L** - Average queue length [requests]
- **W** - Average waiting time [s]
- **μ** (mu) - Service rate [requests/s]
- **c** - Number of servers [requests]

This will generate three key dimensionless coefficients:

- **Occupancy (δ = L/K)** - Queue capacity utilization
- **Stall (σ = W·λ/L)** - Service blocking indicator
- **Efective-Utility (η = K·λ/(c·μ))** - Resource utilization effectiveness

----

Stage 1: Custom Framework Definition
-------------------------------------

Unlike built-in frameworks (PHYSICAL, SOFTWARE), custom frameworks let you define
dimensions specific to your problem domain. Here we create a minimal 2-dimensional
framework for queueing analysis.

Step 1.1: Import Required Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # PyDASA imports
    import pydasa
    from pydasa.workflows.phenomena import AnalysisEngine
    from pydasa.elements.parameter import Variable
    from pydasa.dimensional.vaschy import Schema

    # For visualization and data handling
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import itertools

    print(f"PyDASA Version: {pydasa.__version__}")

Step 1.2: Define Custom Dimensional Framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a schema with only **two** fundamental dimensional units (FDUs):

.. code-block:: python

    # Define simplified FDU list (only T and S)
    fdu_list = [
        {
            "_idx": 0,
            "_sym": "T",
            "_fwk": "CUSTOM",
            "description": "Temporal measurements",
            "_unit": "s",
            "_name": "Time"
        },
        {
            "_idx": 1,
            "_sym": "S",
            "_fwk": "CUSTOM",
            "description": "System capacity and structural resources",
            "_unit": "requests",
            "_name": "Structure"
        }
    ]

    # Create schema
    schema = Schema(_fwk="CUSTOM", _fdu_lt=fdu_list, _idx=0)
    schema._setup_fdus()

    print("=== Simplified Custom Framework Created ===")
    print(f"Framework: {schema.fwk}")
    print(f"Number of FDUs: {len(schema._fdu_lt)}")
    print("\nFundamental Dimensional Units:")
    for fdu in schema._fdu_lt:
        print(f"\t{fdu._sym} ({fdu._name}): {fdu._unit} - {fdu.description}")

**Expected Output:**

.. code-block:: text

    === Simplified Custom Framework Created ===
    Framework: CUSTOM
    Number of FDUs: 2

    Fundamental Dimensional Units:
        T (Time): s - Temporal measurements
        S (Structure): requests - System capacity and structural resources

**Important Notes:**

- Each FDU must have unique ``_idx`` and ``_sym``
- ``_fwk`` must be set to ``"CUSTOM"`` for custom frameworks
- Use descriptive names and units for clarity

Step 1.3: Define Queue Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define **6 variables** using only T and S dimensions:

.. code-block:: python

    # Define simplified variables (only using T and S dimensions)
    variables_dict = {
        # INPUT VARIABLES
        "\\lambda": {
            "_idx": 0,
            "_sym": "\\lambda",
            "_alias": "lambda",
            "_fwk": "CUSTOM",
            "_cat": "IN",
            "_name": "Arrival Rate",
            "description": "Request arrival rate",
            "relevant": True,
            "_dims": "S*T^-1",
            "_units": "req/s",
            "_setpoint": 100.0,
            "_std_setpoint": 100.0,
            "_std_min": 100.0,
            "_std_max": 500.0,
            "_step": 10.0,
        },
        "K": {
            "_idx": 1,
            "_sym": "K",
            "_alias": "K",
            "_fwk": "CUSTOM",
            "_cat": "IN",
            "_name": "Queue Capacity",
            "description": "Maximum system capacity",
            "relevant": True,
            "_dims": "S",
            "_units": "requests",
            "_setpoint": 10.0,
            "_std_setpoint": 10.0,
        },
        
        # OUTPUT VARIABLE
        "L": {
            "_idx": 2,
            "_sym": "L",
            "_alias": "L",
            "_fwk": "CUSTOM",
            "_cat": "OUT",
            "_name": "Queue Length",
            "description": "Average queue length",
            "relevant": True,
            "_dims": "S",
            "_units": "requests",
            "_setpoint": 0.9946,
            "_std_setpoint": 0.9946,
        },
        
        # CONTROL VARIABLES
        "W": {
            "_idx": 3,
            "_sym": "W",
            "_alias": "W",
            "_fwk": "CUSTOM",
            "_cat": "CTRL",
            "_name": "Waiting Time",
            "description": "Average waiting time",
            "relevant": True,
            "_dims": "T",
            "_units": "s",
            "_setpoint": 0.005,
            "_std_setpoint": 0.005,
        },
        "\\mu": {
            "_idx": 4,
            "_sym": "\\mu",
            "_alias": "mu",
            "_fwk": "CUSTOM",
            "_cat": "CTRL",
            "_name": "Service Rate",
            "description": "Service rate per server",
            "relevant": True,
            "_dims": "S*T^-1",
            "_units": "req/s",
            "_setpoint": 400.0,
            "_std_setpoint": 400.0,
            "_std_min": 200.0,
            "_std_max": 1000.0,
        },
        "c": {
            "_idx": 5,
            "_sym": "c",
            "_alias": "c",
            "_fwk": "CUSTOM",
            "_cat": "CTRL",
            "_name": "Servers",
            "description": "Number of parallel servers",
            "relevant": True,
            "_dims": "S",
            "_units": "requests",
            "_setpoint": 1.0,
            "_std_setpoint": 1.0,
            "_std_min": 1.0,
            "_std_max": 4.0,
            "_std_mean": 2.0,
        },
    }

    # Convert to Variable instances
    variables = {
        sym: Variable(**params) for sym, params in variables_dict.items()
    }

    print("=== Variables Defined ===")
    print(f"Total variables: {len(variables)}")
    print(f"Relevant for analysis: {sum(1 for v in variables.values() if v.relevant)}")

**Variable Categories:**

- **INPUT (2)**: λ (arrival rate), K (capacity)
- **OUTPUT (1)**: L (queue length)
- **CONTROL (3)**: W (waiting time), μ (service rate), c (servers)

Step 1.4: Create Analysis Engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create AnalysisEngine
    engine = AnalysisEngine(
        _idx=0,
        _fwk="CUSTOM",
        _schema=schema,
        _name="Simple Queue Analysis",
        description="Simplified M/M/c/K analysis with T and S dimensions only"
    )

    engine.variables = variables

    print("=== Running Dimensional Analysis ===")
    results = engine.run_analysis()

    print(f"\nNumber of Pi Groups: {len(engine.coefficients)}")
    print(f"Coefficients: {list(engine.coefficients.keys())}")

    print("\n=== Dimensionless Coefficients ===")
    for name, coeff in engine.coefficients.items():
        print(f"{name}: {coeff.pi_expr}")

**Expected Output:**

.. code-block:: text

    === Running Dimensional Analysis ===
    
    Number of Pi Groups: 4
    Coefficients: ['\\Pi_{0}', '\\Pi_{1}', '\\Pi_{2}', '\\Pi_{3}']

    === Dimensionless Coefficients ===
    \Pi_{0}: L/K
    \Pi_{1}: W*\lambda/L
    \Pi_{2}: c/K
    \Pi_{3}: \mu/\lambda

Step 1.5: Derive Meaningful Coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Transform the Pi groups into operationally meaningful coefficients:

.. code-block:: python

    # Get Pi coefficient keys
    pi_keys = list(engine.coefficients.keys())

    # Derive meaningful coefficients based on available Pi groups
    print("=== Deriving Operational Coefficients ===")

    # Derive Occupancy Coefficient: δ = Π₀ = L/K
    delta_coeff = engine.derive_coefficient(
        expr=f"{pi_keys[0]}",
        symbol="\\delta",
        name="Occupancy Coefficient",
        description="δ = L/K - Queue occupancy ratio",
        idx=-1
    )

    # Derive Stall Coefficient: σ = Π₁ = W·λ/L
    sigma_coeff = engine.derive_coefficient(
        expr=f"{pi_keys[1]}",
        symbol="\\psi",
        name="Stall Coefficient",
        description="σ = W·λ/L - Service stall/blocking indicator",
        idx=-1
    )

    # Derive Efective-Utility Coefficient: η = Π₂⁻¹·Π₃⁻¹ = K·λ/(c·μ)
    eta_coeff = engine.derive_coefficient(
        expr=f"{pi_keys[2]}**(-1) * {pi_keys[3]}**(-1)",
        symbol="\\eta",
        name="Efective-Utility Coefficient",
        description="η = K·λ/(c·μ) - Resource utilization effectiveness",
        idx=-1
    )

    # Calculate numerical values using stored setpoints
    delta_val = delta_coeff.calculate_setpoint()
    sigma_val = sigma_coeff.calculate_setpoint()
    eta_val = eta_coeff.calculate_setpoint()

    # Occupancy Coefficient: L/K
    print(f"Occupancy: δ=(L/K) = {delta_val:.4f}")

    # Stall Coefficient: σ = W·λ/L
    print(f"Stall: σ = W·λ/L = {sigma_val:.4f}")

    # Efective-Utility Coefficient: η = K·λ/(c·μ)
    print(f"Efective-Utility: η = K·λ/(c·μ) = {eta_val:.4f}")

**Expected Output:**

.. code-block:: text

    === Deriving Operational Coefficients ===
    Occupancy: δ=(L/K) = 0.0995
    Stall: σ = W·λ/L = 0.5027
    Efective-Utility: η = K·λ/(c·μ) = 2.5000

----

Stage 2: Generate Grid-Based Data
----------------------------------

Unlike the Reynolds number example which uses random distributions, queueing
systems require systematic data generation through simulation.

Step 2.1: Import Queue Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from src.queueing import Queue
    import itertools
    import pandas as pd

.. note::
    The ``Queue`` class implements M/M/c/K queueing theory calculations.
    You'll need a queueing library or custom implementation for this step.

Step 2.2: Define Grid Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a systematic grid of configurations:

.. code-block:: python

    print("=== Generating Grid Data ===")

    # Grid parameters
    K_values = [5, 10, 20]          # Capacity values
    c_values = [1.0, 2.0, 4.0]      # Server counts
    mu_values = [200.0, 500.0, 1000.0]  # Service rates

    # Lambda sweep parameters
    lambda_zero = 100.0   # Starting arrival rate
    lambda_step = 10.0    # Increment
    RHO_THLD = 0.95      # Utilization threshold

    # Generate configurations (3 × 3 × 3 = 27 configurations)
    cfg_lt = list(itertools.product(K_values, c_values, mu_values))
    print(f"Total configurations: {len(cfg_lt)}")

**Expected Output:**

.. code-block:: text

    === Generating Grid Data ===
    Total configurations: 27

Step 2.3: Generate Queue Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each configuration, sweep arrival rate until utilization threshold:

.. code-block:: python

    # Create DataFrame
    cols = list(variables.keys())
    data_df = pd.DataFrame(columns=cols)

    # Generate data
    for idx, (K, c, mu) in enumerate(cfg_lt, 1):
        print(f"\t-*- Config {idx}/{len(cfg_lt)}: K={K}, c={c}, μ={mu} -*-")
        lambda_t = lambda_zero
        rho_t = 0.0
        
        while rho_t < RHO_THLD:
            # Calculate queue metrics for M/M/c/K model
            q = Queue("M/M/s/K", lambda_t, mu, int(c), K)
            q.calculate_metrics()
            
            # Order must match cols = ["\\lambda", "K", "L", "W", "\\mu", "c"]
            data_t = [lambda_t, K, q.avg_len, q.avg_wait, mu, c]
            data_df.loc[len(data_df)] = data_t
            
            rho_t = q.rho
            lambda_t += lambda_step

    print(f"Generated {len(data_df)} data points")

    # Add data to variables
    data = data_df.to_dict(orient="list")
    for sym, var in variables.items():
        if sym in data:
            var.data = data[sym]

    engine.variables = variables
    print("Data injected into PyDASA engine!")

**Expected Output:**

.. code-block:: text

    === Generating Grid Data ===
    Total configurations: 27
        -*- Config 1/27: K=5, c=1.0, μ=200.0 -*-
        -*- Config 2/27: K=5, c=1.0, μ=500.0 -*-
    ...
        -*- Config 27/27: K=20, c=4.0, μ=1000.0 -*-
    Generated 3150 data points
    Data injected into PyDASA engine!

**Critical Note:**

The order in ``data_t`` must **exactly match** the column order in ``cols``.
Misalignment will cause incorrect analysis results.

Step 2.4: Inspect Generated Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Display statistical summary
    print(data_df.describe())

----

Stage 3: Monte Carlo Simulation with Grid Data
-----------------------------------------------

Run Monte Carlo simulation using the systematically generated data.

Step 3.1: Create Simulation Handler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pydasa.workflows.practical import MonteCarloSimulation

    print("=== Running Monte Carlo Simulation ===")

    mc_grid = MonteCarloSimulation(
        _idx=0,
        _fwk="CUSTOM",
        _schema=schema,
        _name="Simple Grid Queue Analysis",
        _cat="DATA",
        _experiments=len(data_df),
        _variables=engine.variables,
        _coefficients=engine.coefficients
    )

    mc_grid.run_simulation(iters=len(data_df))

    print(f"Simulation complete: {mc_grid.experiments} experiments")
    print(f"Results for: {list(mc_grid.results.keys())}")

**Expected Output:**

.. code-block:: text

    === Running Monte Carlo Simulation ===
    Simulation complete: 3150 experiments
    Results for: ['\\Pi_{0}', '\\Pi_{1}', '\\Pi_{2}', '\\Pi_{3}', '\\delta', '\\psi', '\\eta']

Step 3.2: Extract Simulation Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Extract simulation data
    all_keys = list(mc_grid.simulations.keys())
    derived_keys = [k for k in all_keys if not k.startswith('\\Pi_')]

    print("=== Extracting Simulation Results ===")
    pi_data = {}
    for pi_key in derived_keys:
        print(f"Processing simulation: {pi_key}")
        pi_sim_obj = mc_grid.get_simulation(pi_key)
        pi_results = pi_sim_obj.extract_results()
        for sym, var in pi_sim_obj.variables.items():
            pi_data[sym] = var.data
        pi_data[pi_key] = pi_results[pi_key]

    # Get data arrays
    print("\n=== Simulation Results Details ===")
    if len(derived_keys) > 0:
        delta_sim = pi_data[derived_keys[0]]
        sigma_sim = pi_data[derived_keys[1]]
        eta_sim = pi_data[derived_keys[2]]
        lambda_data = np.array(pi_data["\\lambda"])
        mu_data = np.array(pi_data["\\mu"])
        c_data = np.array(pi_data["c"])
        K_data = np.array(pi_data["K"])
        
        print(f"Occupancy (δ): Mean = {np.mean(delta_sim):.4e}, Range = [{np.min(delta_sim):.4e}, {np.max(delta_sim):.4e}]")
        print(f"Stall (σ): Mean = {np.mean(sigma_sim):.4e}, Range = [{np.min(sigma_sim):.4e}, {np.max(sigma_sim):.4e}]")
        print(f"Effective-Yield (η): Mean = {np.mean(eta_sim):.4e}, Range = [{np.min(eta_sim):.4e}, {np.max(eta_sim):.4e}]")

**Expected Output:**

.. code-block:: text

    === Extracting Simulation Results ===
    Processing simulation: \delta
    Processing simulation: \psi
    Processing simulation: \eta

    === Simulation Results Details ===
    Occupancy (δ): Mean = 4.2156e-01, Range = [9.9463e-02, 9.5238e-01]
    Stall (σ): Mean = 1.4532e+00, Range = [5.0271e-01, 7.1250e+00]
    Effective-Yield (η): Mean = 7.5463e-01, Range = [6.2500e-02, 2.5000e+00]

----

Stage 4: Advanced Visualization
--------------------------------

Create comprehensive visualizations to understand dimensionless behavior
across multiple configurations.

Step 4.1: 3D Yoly Diagram with 2D Projections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a 2×2 grid showing 3D space and three 2D projections:

.. code-block:: python

    # Create comprehensive Yoly diagram with 3D and 2D projections in a 2x2 grid
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(20, 16), facecolor="white")

    # Define grid specification for 2x2 layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Create axes: [0,0] is 3D, others are 2D
    axes = [
        [fig.add_subplot(gs[0, 0], projection="3d"), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ]

    # Auxiliary lists for plot configuration
    plot_types = ["3D", "2D", "2D", "2D"]
    plot_titles = [
        r"3D Space: $\boldsymbol{\delta}$ vs $\boldsymbol{\psi}$ vs $\boldsymbol{\eta}$",
        r"2D Plane: $\boldsymbol{\delta}$ vs $\boldsymbol{\psi}$",
        r"2D Plane: $\boldsymbol{\delta}$ vs $\boldsymbol{\eta}$",
        r"2D Plane: $\boldsymbol{\psi}$ vs $\boldsymbol{\eta}$"
    ]

    x_labels = [
        r"Occupancy ($\boldsymbol{\delta}$)",
        r"Occupancy ($\boldsymbol{\delta}$)",
        r"Occupancy ($\boldsymbol{\delta}$)",
        r"Stall ($\boldsymbol{\psi}$)"
    ]

    y_labels = [
        r"Stall ($\boldsymbol{\psi}$)",
        r"Stall ($\boldsymbol{\psi}$)",
        r"Effective-Yield ($\boldsymbol{\eta}$)",
        r"Effective-Yield ($\boldsymbol{\eta}$)"
    ]

    z_labels = [r"Effective-Yield ($\boldsymbol{\eta}$)", None, None, None]

    # Data pairs for each subplot
    data_pairs = [
        (delta_sim, sigma_sim, eta_sim),    # 3D plot
        (delta_sim, sigma_sim, None),       # delta vs sigma
        (delta_sim, eta_sim, None),         # delta vs eta
        (sigma_sim, eta_sim, None)          # sigma vs eta
    ]

Step 4.2: Configure Color and Marker Mapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use color for server count and markers for service rate:

.. code-block:: python

    # Color map for server count (c) and marker map for service rates (μ)
    c_data = np.array(variables["c"].data)
    mu_data = np.array(variables["\\mu"].data)
    K_data = np.array(variables["K"].data)
    unique_c = np.unique(c_data)
    unique_mu = np.unique(mu_data)
    unique_K = np.unique(K_data)

    # Colors for servers: red (1), orange (2), green (4)
    color_map = {1.0: "red", 2.0: "orange", 4.0: "green"}
    # Markers for service rate: triangle (slow), square (mid), circle (fast)
    marker_map = {200: "^", 500: "s", 1000: "o"}

Step 4.3: Plot Data with Masks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Iterate over 2x2 grid
    plot_idx = 0
    for row in range(2):
        for col in range(2):
            ax = axes[row][col]
            plot_type = plot_types[plot_idx]
            x_data, y_data, z_data = data_pairs[plot_idx]

            # Set white background for subplot
            ax.set_facecolor("white")

            # Track which (c, μ) combinations have been labeled for legend
            labeled_combinations = set()

            # Plot data points grouped by server count (c), service rate (μ), and capacity (K)
            for c_val in unique_c:
                for mu_val in unique_mu:
                    for K_val in unique_K:
                        # Create mask for this specific combination
                        mask = (np.abs(c_data - c_val) < 0.1) & (np.abs(mu_data - mu_val) < 0.1) & (np.abs(K_data - K_val) < 0.1)
                        if not np.any(mask):
                            continue

                        # Create label only once per (c, μ) combination
                        combo_key = (c_val, mu_val)
                        if combo_key not in labeled_combinations:
                            label = f"c={int(c_val)}, μ={int(mu_val)}"
                            labeled_combinations.add(combo_key)
                        else:
                            label = None  # No label for subsequent K values

                        color = color_map.get(c_val, "gray")
                        marker = marker_map.get(mu_val, "o")

                        if plot_type == "3D":
                            # 3D scatter plot
                            ax.scatter(x_data[mask], y_data[mask], z_data[mask],
                                       c=color, marker=marker, s=30, alpha=0.6,
                                       edgecolors="grey", linewidths=0.1,
                                       label=label)
                            # Add K value label at median position
                            mask_indices = np.where(mask)[0]
                            if len(mask_indices) > 0:
                                mid_idx = mask_indices[len(mask_indices)//2]
                                ax.text(x_data[mid_idx], y_data[mid_idx], z_data[mid_idx],
                                        f"K={int(K_val)}", fontsize=8, color="black",
                                        fontweight="bold", alpha=0.8)
                        else:
                            # 2D scatter plot
                            ax.scatter(x_data[mask], y_data[mask],
                                       c=color, marker=marker, s=30, alpha=0.6,
                                       edgecolors="grey", linewidths=0.1,
                                       label=label)

            # Apply plot-specific styling
            if plot_type == "3D":
                ax.set_zlabel(z_labels[plot_idx], fontsize=12, fontweight="bold", color="black")
                ax.view_init(elev=30, azim=110)
                ax.grid(True, color="dimgray", linewidth=1, linestyle="--", alpha=0.9)
            else:
                ax.grid(True, alpha=0.8, color="dimgray", linewidth=1.0, linestyle="--")

            # Set labels and title
            ax.set_xlabel(x_labels[plot_idx], fontsize=12, fontweight="bold", color="black")
            ax.set_ylabel(y_labels[plot_idx], fontsize=12, fontweight="bold", color="black")
            ax.set_title(plot_titles[plot_idx], fontsize=14, fontweight="bold", pad=10, color="black")
            ax.legend(loc="best", fontsize=10, framealpha=0.9)

            plot_idx += 1

    # Add main title
    fig.suptitle("Comprehensive Yoly Diagram: M/M/c/K Queue Analysis\n",
                 fontsize=16, fontweight="bold", y=0.98, color="black")
    plt.show()

**Visualization Shows:**

- **3D Space**: Complete relationship between all three coefficients
- **2D Projections**: Detailed views of coefficient pairs
- **Color Coding**: Server count (red=1, orange=2, green=4)
- **Markers**: Service rate (triangle=slow, square=mid, circle=fast)
- **Labels**: Capacity (K) values annotated on clusters

----

Tips and Best Practices
------------------------

**Custom Framework Design**

When creating custom frameworks, choose FDUs that:

- Are truly fundamental to your domain
- Cannot be expressed in terms of each other
- Capture the essential physics/behavior
- Keep the number minimal (2-4 FDUs typically sufficient)

**Dimension String Format**

Use proper syntax for custom dimensions:

.. code-block:: python

    # Good: Rate dimension [S/T]
    "_dims": "S*T^-1"
    
    # Good: Dimensionless
    "_dims": "1"
    
    # Bad: Mixing frameworks
    "_dims": "M*S*T^-1"  # Don't mix PHYSICAL (M) with CUSTOM (S)

**Data Generation Strategies**

For systematic analysis:

1. **Grid Search**: Exhaustive coverage of parameter space (as shown)
2. **Latin Hypercube**: Efficient sampling for large parameter spaces
3. **Adaptive Sampling**: Focus on regions of interest

**Variable Ordering**

Always ensure data alignment:

.. code-block:: python

    # Get column order
    cols = list(variables.keys())  # ["\\lambda", "K", "L", "W", "\\mu", "c"]
    
    # MUST match this order when adding data
    data_t = [lambda_t, K, q.avg_len, q.avg_wait, mu, c]

**Coefficient Derivation**

Use mathematical expressions to combine Pi groups:

.. code-block:: python

    # Inversion
    expr=f"{pi_keys[0]}**(-1)"
    
    # Product
    expr=f"{pi_keys[0]} * {pi_keys[1]}"
    
    # Complex combination
    expr=f"{pi_keys[2]}**(-1) * {pi_keys[3]}**(-1)"

Common Pitfalls
---------------

**1. Framework Mismatch**

.. code-block:: python

    # Wrong: Using PHYSICAL framework with custom schema
    engine = AnalysisEngine(_fwk="PHYSICAL", _schema=custom_schema)
    
    # Correct: Framework must match schema
    engine = AnalysisEngine(_fwk="CUSTOM", _schema=custom_schema)

**2. Missing Schema Setup**

.. code-block:: python

    # Wrong: Forgot to setup FDUs
    schema = Schema(_fwk="CUSTOM", _fdu_lt=fdu_list, _idx=0)
    # Missing: schema._setup_fdus()
    
    # Correct: Always call setup
    schema = Schema(_fwk="CUSTOM", _fdu_lt=fdu_list, _idx=0)
    schema._setup_fdus()

**3. Data Alignment Error**

.. code-block:: python

    # Wrong order causes incorrect analysis
    data_t = [lambda_t, K, mu, q.avg_wait, q.avg_len, c]
    # This assigns mu to L, q.avg_wait to W, q.avg_len to mu!
    
    # Correct order matching ["\\lambda", "K", "L", "W", "\\mu", "c"]
    data_t = [lambda_t, K, q.avg_len, q.avg_wait, mu, c]

**4. Forgetting Variable Data Injection**

.. code-block:: python

    # Wrong: Data generated but not injected
    data_df = pd.DataFrame(...)
    mc_grid = MonteCarloSimulation(..., _variables=engine.variables)
    # Variables don't have data yet!
    
    # Correct: Inject data before simulation
    data = data_df.to_dict(orient="list")
    for sym, var in variables.items():
        if sym in data:
            var.data = data[sym]
    engine.variables = variables

Summary
-------

**Custom Framework Workflow:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Stage
     - Key Actions
   * - **1. Framework**
     - Define custom FDUs, create schema, define variables
   * - **2. Analysis**
     - Run AnalysisEngine, derive meaningful coefficients
   * - **3. Data Generation**
     - Generate systematic grid data from domain model
   * - **4. Simulation**
     - Run Monte Carlo with grid data, extract results
   * - **5. Visualization**
     - Create comprehensive multi-dimensional plots

**Key Differences from Standard Workflows:**

1. **Custom FDUs**: Define your own fundamental dimensions
2. **Domain-Specific**: Tailor to non-traditional engineering domains
3. **Grid Data**: Systematic parameter sweeps vs. random distributions
4. **Specialized Viz**: Multi-configuration comparative analysis

**When to Use Custom Frameworks:**

- Analyzing software/cyber systems (requests, packets, bits)
- Economic/business problems (transactions, currency, time)
- Biological systems (cells, molecules, reactions)
- Any domain where PHYSICAL framework doesn't apply naturally

----

Next Steps
----------

Now that you've mastered custom frameworks, explore:

- Create multi-dimensional frameworks (3+ FDUs)
- Integrate with domain-specific simulation tools
- Compare custom vs. built-in framework results
- Develop framework-specific visualization templates

.. seealso::

    **Full Working Examples:**

    **Notebooks:**
    
    - :download:`Online Tutorial Notebook <../../_static/examples/notebooks/PyDASA-Online-Tutorial.ipynb>`
    - :download:`Online Customization Notebook <../../_static/examples/notebooks/PyDASA-Online-Custom.ipynb>`
    
    **Code:**
    
    - :download:`reynolds_simple.py <../../_static/examples/code/reynolds_simple.py>`
    - :download:`queueing.py <../../_static/examples/code/queueing.py>`

**Documentation and User Guides:**

To understand **PyDASA** capabilities and internal technical details check:

- :doc:`../features/index` - Available features and workflows.
- :doc:`../examples/index` - More practical examples using PyDASA.
- :doc:`../../autoapi/index` - Complete API reference.

**Questions or Issues?**

Visit the `GitHub Issues Page <https://github.com/DASA-Design/PyDASA/issues>`_ for support.