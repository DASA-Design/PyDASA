Tutorial
========

This tutorial will guide you through performing dimensional analysis with PyDASA,
from basic concepts to advanced techniques.

.. contents:: Table of Contents
    :local:
    :depth: 2

Introduction to Dimensional Analysis
-------------------------------------

Dimensional analysis is a mathematical technique used to study the relationships
between physical quantities by identifying their fundamental dimensions (length, mass, time, etc.).
The Buckingham Pi theorem allows us to reduce complex physical relationships into
simpler dimensionless forms.

Traditional FDU Framework
--------------------------

In classical dimensional analysis, we work with fundamental dimensions:

* **F** - Force
* **D** - Distance (Length)
* **U** - Unity (Time)

Example: Pendulum Period
^^^^^^^^^^^^^^^^^^^^^^^^^

Let's analyze the period of a simple pendulum using the traditional FDU framework.

**Physical Setup:**

A pendulum's period ``T`` depends on:
- Length of the string ``L``
- Mass of the bob ``m``
- Gravitational acceleration ``g``

**Step 1: Define Variables with FDU Dimensions**

.. code-block:: python

    from pydasa.core.fundamental import Dimension
    from pydasa.core.parameter import Variable
    from pydasa.handler.phenomena import Solver

    # Define fundamental dimensions using FDU framework
    # Force [F], Distance [D], Unity (Time) [U]
    F = Dimension(force=1)
    D = Dimension(length=1)
    U = Dimension(time=1)

    # Note: In FDU framework, mass is derived from F, D, U
    # [m] = F·U²/D

    # Define variables
    period = Variable(
        name="T",
        dimension=U,  # Period has dimension of time [U]
        value=2.0,    # seconds
        description="Period of pendulum"
    )

    length = Variable(
        name="L",
        dimension=D,  # Length [D]
        value=1.0,    # meters
        description="Length of pendulum string"
    )

    mass = Variable(
        name="m",
        dimension=F * U**2 / D,  # Mass [F·U²/D]
        value=0.5,    # kg
        description="Mass of pendulum bob"
    )

    gravity = Variable(
        name="g",
        dimension=D / U**2,  # Acceleration [D/U²]
        value=9.81,   # m/s²
        description="Gravitational acceleration"
    )

**Step 2: Solve for Dimensionless Numbers**

.. code-block:: python

    # Create solver
    solver = Solver()

    # Add variables
    solver.add_variables([period, length, mass, gravity])

    # Solve using Buckingham Pi theorem
    result = solver.solve()

    # Display dimensionless Pi groups
    for i, pi in enumerate(result.pi_groups, 1):
        print(f"π{i} = {pi}")

**Expected Output:**

.. code-block:: text

    π1 = T·√(g/L)

This tells us that ``T·√(g/L)`` is dimensionless, which means:

.. math::

    T \propto \sqrt{\frac{L}{g}}

The mass ``m`` doesn't appear in the final relationship!

Custom Dimension Framework
---------------------------

PyDASA also supports custom dimension frameworks beyond FDU. Let's use the
more common **MLT** (Mass-Length-Time) framework.

Example: Drag Force on a Sphere
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We want to find the drag force on a sphere moving through a fluid.

**Physical Setup:**

The drag force ``F_d`` depends on:
- Fluid density ``ρ``
- Velocity ``v``
- Sphere diameter ``d``
- Fluid viscosity ``μ``

**Step 1: Define Custom MLT Framework**

.. code-block:: python

    from pydasa.core.fundamental import Dimension
    from pydasa.core.parameter import Variable
    from pydasa.handler.phenomena import Solver

    # Define fundamental dimensions using MLT framework
    M = Dimension(mass=1)      # Mass
    L = Dimension(length=1)    # Length
    T = Dimension(time=1)      # Time

    # Define variables with MLT dimensions
    drag_force = Variable(
        name="F_d",
        dimension=M * L / T**2,  # Force [M·L/T²]
        value=10.0,
        description="Drag force"
    )

    density = Variable(
        name="ρ",
        dimension=M / L**3,  # Density [M/L³]
        value=1000.0,        # kg/m³ (water)
        description="Fluid density"
    )

    velocity = Variable(
        name="v",
        dimension=L / T,  # Velocity [L/T]
        value=2.0,        # m/s
        description="Sphere velocity"
    )

    diameter = Variable(
        name="d",
        dimension=L,  # Length [L]
        value=0.1,    # meters
        description="Sphere diameter"
    )

    viscosity = Variable(
        name="μ",
        dimension=M / (L * T),  # Dynamic viscosity [M/(L·T)]
        value=0.001,            # Pa·s
        description="Fluid viscosity"
    )

**Step 2: Solve for Dimensionless Numbers**

.. code-block:: python

    # Create solver
    solver = Solver()

    # Add all variables
    solver.add_variables([drag_force, density, velocity, diameter, viscosity])

    # Solve
    result = solver.solve()

    # Display Pi groups
    for i, pi in enumerate(result.pi_groups, 1):
        print(f"π{i} = {pi}")
        print(f"   Value: {pi.evaluate():.4f}")

**Expected Output:**

.. code-block:: text

    π1 = F_d / (ρ·v²·d²)     # Drag coefficient
        Value: 0.2500
    
    π2 = (ρ·v·d) / μ          # Reynolds number
        Value: 200000.0000

This gives us two important dimensionless numbers:

1. **Drag Coefficient**: ``C_d = F_d / (ρ·v²·d²)``
2. **Reynolds Number**: ``Re = (ρ·v·d) / μ``

The relationship becomes:

.. math::

   C_d = f(Re)

Advanced Example: Thermal System
---------------------------------

Let's analyze a thermal system with even more dimensions.

**Physical Setup:**

Heat transfer ``Q`` in a convection system depends on:
- Temperature difference ``ΔT``
- Surface area ``A``
- Thermal conductivity ``k``
- Heat transfer coefficient ``h``
- Fluid velocity ``v``

**Step 1: Define MLTT Framework (Mass-Length-Time-Temperature)**

.. code-block:: python

    from pydasa.core.fundamental import Dimension
    from pydasa.core.parameter import Variable
    from pydasa.handler.phenomena import Solver

    # Extended framework with temperature
    M = Dimension(mass=1)
    L = Dimension(length=1)
    T = Dimension(time=1)
    Θ = Dimension(temperature=1)  # Temperature

    # Define variables
    heat_transfer = Variable(
        name="Q",
        dimension=M * L**2 / T**3,  # Power [M·L²/T³]
        value=1000.0,
        description="Heat transfer rate"
    )

    temp_diff = Variable(
        name="ΔT",
        dimension=Θ,  # Temperature [Θ]
        value=50.0,
        description="Temperature difference"
    )

    area = Variable(
        name="A",
        dimension=L**2,  # Area [L²]
        value=0.5,
        description="Surface area"
    )

    thermal_cond = Variable(
        name="k",
        dimension=M * L / (T**3 * Θ),  # Thermal conductivity
        value=0.6,
        description="Thermal conductivity"
    )

    htc = Variable(
        name="h",
        dimension=M / (T**3 * Θ),  # Heat transfer coefficient
        value=50.0,
        description="Heat transfer coefficient"
    )

    velocity = Variable(
        name="v",
        dimension=L / T,  # Velocity [L/T]
        value=1.5,
        description="Fluid velocity"
    )

**Step 2: Solve and Analyze**

.. code-block:: python

    # Create and configure solver
    solver = Solver()
    solver.add_variables([
        heat_transfer, temp_diff, area,
        thermal_cond, htc, velocity
    ])

    # Solve
    result = solver.solve()

    # Display results
    print(f"Number of Pi groups: {len(result.pi_groups)}")
    print("\nDimensionless numbers:")
    for i, pi in enumerate(result.pi_groups, 1):
        print(f"\nπ{i}:")
        print(f"  Expression: {pi}")
        print(f"  Value: {pi.evaluate():.6f}")
        print(f"  Physical meaning: {pi.description}")

**Expected Output:**

.. code-block:: text

    Number of Pi groups: 2

    Dimensionless numbers:

    π1:
        Expression: Q / (h·A·ΔT)
        Value: 0.800000
        Physical meaning: Normalized heat transfer

    π2:
        Expression: (h·L) / k
        Value: 41.666667
        Physical meaning: Biot number

Working with Results
--------------------

**Accessing Pi Group Information**

.. code-block:: python

    # Get specific Pi group
    pi1 = result.pi_groups[0]

    # Get constituent variables
    print(f"Variables in π1: {pi1.variables}")

    # Get exponents
    for var, exp in pi1.exponents.items():
        print(f"{var.name}: {exp}")

    # Evaluate with different values
    new_value = pi1.evaluate(overrides={'Q': 2000.0})
    print(f"π1 with Q=2000: {new_value}")

**Sensitivity Analysis**

.. code-block:: python

    from pydasa.analysis.scenario import DimSensitivity

    # Create sensitivity analyzer
    sensitivity = DimSensitivity(result.pi_groups[0])

    # Define variable bounds for analysis
    bounds = {
        'Q': (500, 1500),
        'h': (20, 100),
        'A': (0.3, 0.8),
        'ΔT': (30, 70)
    }

    # Run sensitivity analysis
    sensitivity.set_bounds(bounds)
    sensitivity_results = sensitivity.analyze()

    # Display results
    print("\nSensitivity indices:")
    for var, index in sensitivity_results.items():
        print(f"{var}: {index:.4f}")

**Monte Carlo Simulation**

.. code-block:: python

    from pydasa.analysis.simulation import MonteCarloSimulator

    # Create simulator
    simulator = MonteCarloSimulator(result.pi_groups[0])

    # Define distributions for variables
    distributions = {
        'Q': {'type': 'normal', 'mean': 1000, 'std': 100},
        'h': {'type': 'uniform', 'min': 40, 'max': 60},
        'A': {'type': 'normal', 'mean': 0.5, 'std': 0.05},
        'ΔT': {'type': 'uniform', 'min': 45, 'max': 55}
    }

    # Run simulation
    results = simulator.run(
        n_samples=10000,
        distributions=distributions
    )

    # Analyze results
    print(f"\nMonte Carlo Results:")
    print(f"Mean: {results.mean():.6f}")
    print(f"Std Dev: {results.std():.6f}")
    print(f"95% CI: [{results.quantile(0.025):.6f}, {results.quantile(0.975):.6f}]")

Best Practices
--------------

1. **Choose the Right Framework**

   - Use **FDU** for mechanics problems in engineering contexts
   - Use **MLT** for general physics problems
   - Extend with additional dimensions (Θ, I, N) as needed

2. **Validate Dimensions**

    .. code-block:: python

        # PyDASA automatically validates dimensional consistency
        try:
            invalid_var = Variable("x", M + L)  # This will raise an error
        except ValueError as e:
            print(f"Dimension error: {e}")

3. **Document Your Variables**

    .. code-block:: python

        velocity = Variable(
            name="v",
            dimension=L / T,
            value=10.0,
            description="Fluid velocity at inlet",
            units="m/s"  # Optional but recommended
        )

4. **Use Meaningful Variable Names**

    .. code-block:: python

        # Good
        reynolds_number = pi_groups[0]
        
        # Better
        Re = pi_groups[0]
        Re.name = "Reynolds Number"
        Re.symbol = "Re"

Next Steps
----------

Now that you understand the basics, explore:

- :doc:`../user_guide/dimensional_analysis` - Deep dive into theory
- :doc:`../examples/basics` - More practical examples
- :doc:`../user_guide/simulation` - Advanced Monte Carlo techniques
- :doc:`../user_guide/sensitivity` - Sensitivity analysis methods

.. seealso::

    - :doc:`quickstart` - Quick reference guide
    - :doc:`installation` - Installation options
    - :doc:`../api/index` - Complete API documentation