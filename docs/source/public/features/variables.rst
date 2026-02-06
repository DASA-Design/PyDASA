Variables
=================

Overview
---------

Variables are the main entities in dimensional analysis, representing measurable or calculable quantities within a system. In **PyDASA**, the ``Variable`` class implements a composite architecture that integrates four complementary perspectives: **Conceptual**, **Symbolic**, **Numerical**, and **Statistical** to support flexible dimensional modeling, sensitivity analysis, and numerical simulation.

This multi-perspective design addresses Görtler's principles (Measurability, Consistency, Clarity) while balancing rigor with flexibility. Critically, **not all attributes must be defined**; users specify only the features relevant to their analysis, supporting customized dimensional models ranging from purely symbolic expressions goping into fully nprobabilistic simulations and practical experimental validations.


Analytical Flexibility
----------------------

This composite architecture provides support for three primary analytical workflows:

1. **Dimensional Modeling** uses the Conceptual and Symbolic perspectives only. Practitioners can define only ``_cat``, ``_dims``, ``_units``, and ``_schema`` to construct dimensional matrices and derive dimensionless groups. Optional numerical bounds (``_setpoint`` and ``_std_setpoint``) can validate feasibility without constraining the model.
2. **Sensitivity Analysis** uses the Conceptual, Symbolic, and Numerical perspectives. Defining ``_min``, ``_max``, and ``_step`` for INPUT variables allows systematic exploration of how input variations affect dimensionless coefficients, with the framework automatically discretizing ranges and generating parameter sweeps for comprehensive sensitivity mapping.
3. **Monte Carlo Simulation** uses all four perspectives (Statistical optional for INPUT variables). Fully specified variables with distributions (``_dist_type``, ``_dist_params``) support stochastic or real experimental sampling, while OUTPUT variables may omit distributions as their uncertainty propagates from INPUT dependencies via ``_depends`` relationships.

This staged approach allows users to invest effort proportional to analytical depth, avoiding unnecessary complexity for simpler analyses.

Customizable Capabilities
---------------------------

**PyDASA's** Variable design prioritizes flexibility:

1. **Partial Specification** Define only required attributes; unspecified features default gracefully.
2. **Schema Independence** Variables work with any framework (``PHYSICAL``, ``COMPUTATION``, ``SOFTWARE``, ``CUSTOM``).
3. **Custom Distributions** User-defined ``_dist_func`` enables arbitrary sampling logic beyond standard distributions
4. **Dynamic Dependencies** OUTPUT variables recalculate from INPUT samples automatically during simulation.
5. **Standardization Control** Override default unit conversions with custom ``_std_units`` mappings.

This flexibility allows researchers to adapt **PyDASA** to domain-specific needs from the traditional aerodynamic scaling laws (``PHYSICAL``) to the more niche algorithm complexity analysis (``COMPUTATION``) without modifying core library code.

Practical Example
-----------------

Consider modeling computational performance with three variables:

.. code-block:: python

    from pydasa import Variable, Schema

    schema = Schema(_fwk="COMPUTATION")  # Time, Data, Operations
    print(schema)

    # INPUT: Problem size (minimal specification for dimensional model)
    N = Variable(
        _name="Problem Size",
        _sym="N",
        _cat="INPUT",
        _dims="S",  # Data dimension
        _units="elements",
        _schema=schema
    )

    # INPUT with numerical bounds for sensitivity analysis
    T = Variable(
        _name="Time Budget",
        _sym="T",
        _cat="INPUT",
        _dims="T",
        _units="seconds",
        _min=1.0,
        _max=10.0,
        _schema=schema
    )

    # INPUT with full distribution for Monte Carlo
    throughput = Variable(
        _name="Throughput",
        _sym="R",
        _cat="INPUT",
        _dims="S*T^-1",
        _units="MB/s",
        _dist_type="normal",
        _dist_params={"mean": 100.0, "std": 10.0},
        _schema=schema
    )

    # OUTPUT: Depends on other variables (distribution computed automatically)
    latency = Variable(
        _name="Latency",
        _sym="L",
        _cat="OUTPUT",
        _dims="T",
        _units="milliseconds",
        _depends=["N", "R"],  # Function of problem size and throughput
        _schema=schema
    )

This example illustrates a progressive enhancement scenario: dimensional modeling requires only symbolic attributes, sensitivity analysis adds numerical bounds, and Monte Carlo simulation adds probabilistic specifications; all within the same ``Variable`` class.

Composite Architecture
----------------------

The ``Variable`` class composes four modules, each one answering a fundamental question about the variable's role in the dimensional analysis process:

1. The **Conceptual Perspective** (``conceptual.py``) answers *"What IS this variable?"* by defining its identity and classification within the dimensional framework:
    - **Category** (``_cat``): Classifies variable as INPUT (``IN``), OUTPUT (``OUT``), or CONTROL (``CTRL``).
    - **Schema Reference** (``_schema``): Links to the dimensional framework (*PHYSICAL*, *COMPUTATION*, *SOFTWARE*, *CUSTOM*).
    - **Relevance Flag** (``relevant``): Marks variables for inclusion in specific analyses (``True`` or ``False``).

This perspective filters variables by role (INPUT, OUTPUT, CTRL), customizing precedence in dimensional matrix and solution.

2. The **Symbolic Perspective** (``symbolic.py``) answers *"How do we WRITE this variable?"* by managing its mathematical representation and dimensional properties:
    - **Dimensional Expression** (``_dims``): Defines *FDU* formula notation (e.g., ``L*T^-1`` for velocity).
    - **Units** (``_units``): Specifies measurement units in original scale (e.g., ``m/s``, ``km/h``).
    - **Dimensional Column** (``_dim_col``): Provides vector representation for dimensional matrix operations (e.g., ``[1, 0, -1]`` for ``L*T^-1``).
    - **Standardized Forms** (``_std_dims``, ``_sym_exp``): Stores canonical expressions for consistency across analyses.
    - **Standardized Units** (``_std_units``): Converts to *IS* or framework-specific standard units (e.g., ``m/s`` from ``km/h``).

This perspective enables dimensional matrix construction, homogeneity validation, and coefficient derivation via Buckingham Pi-Theorem.

3. The **Numerical Perspective** (``numerical.py``) answers *"What VALUES can this variable take?"* by specifying value ranges and discretization properties for computational analysis:
    - **Bounds** (``_min``, ``_max``, ``_mean``, ``_median``, ``_dev``): Define statistical measures in original units (e.g., min=10, max=100, mean=55).
    - **Standardized Bounds** (``_std_min``, ``_std_max``, ``_std_mean``, ``_std_median``, ``_std_dev``): Store values in *SI* or framework-specific standard units.
    - **Setpoints** (``_setpoint``, ``_std_setpoint``): Specify fixed reference values for specific operational scenarios (e.g., nominal operating point).

This perspective supports sensitivity analysis by defining feasible operating ranges, facilitating grid-based exploration of the design space and systematic parameter sweeps.

.. warning::
    **IMPORTANT**: As of January 2026, numerical standardizations (e.g., ``_setpoint`` to ``_std_setpoint``) must be manually provided by users. AND sensitivity analysis and simulations require these standardized values to proceed correctly.

4. The **Statistical Perspective** (``statistical.py``) answers *"How do we MODEL uncertainty?"* by defining probabilistic distributions and dependencies for Monte Carlo simulation:
    - **Distribution Type** (``_dist_type``): Specifies probability distribution families (``uniform``, ``normal``, ``triangular``, ``exponential``, ``lognormal``, ``custom``).
    - **Distribution Parameters** (``_dist_params``): Defines distribution-specific configuration (e.g., ``{"mean": 100, "std": 10}`` for normal, ``{"min": 0, "max": 1}`` for uniform).
    - **Distribution Function** (``_dist_func``): Provides user-defined callable for custom sampling logic or complex dependencies between variables.
    - **Dependencies** (``_depends``): Lists variable names this variable depends on (e.g., ``["F", "m"]`` for ``a = F/m``), supporting automatic OUTPUT calculations.

This perspective supports Monte Carlo simulation through uncertainty quantification and propagation, facilitating stochastic analysis of dimensional model behavior.