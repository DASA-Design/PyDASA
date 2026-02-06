Dimensional Analysis
====================

Overview
---------

The ``AnalysisEngine`` class provides high-level interface to dimensional analysis workflows in **PyDASA**, abstracting matrix operations with customization and data integration. Orchestrates dimensional analysis from schema configuration through coefficient generation with serialization for pipeline integration.

The ``AnalysisEngine`` simplifies dimensional analysis through three core features:

1. **Defines Frameworks**: Configures dimensional domains with custom FDUs for specific problem contexts.
2. **Automates Workflows**: Handles matrix creation, solving, and coefficient generation via method calls.
3. **Serializes Data**: Built-in dictionary serialization (``to_dict()``, ``from_dict()``) for JSON export/import and data persistence.

Reynolds Number (:math:`Re`) Dimensional Analysis Example
-----------------------------------------------------------

This example demonstrates deducing the Reynolds number using a custom framework with selective variable relevance and data export capabilities:

.. code-block:: python

    from pydasa import Variable, Schema, AnalysisEngine
    from pydasa.dimensional.fundamental import Dimension
    import json
    import os

    # Define custom framework (T, M, L only - typical for fluid mechanics)
    custom_fdus = [
        Dimension(_idx=0, _sym="T", _unit="s", _name="Time"),
        Dimension(_idx=1, _sym="M", _unit="kg", _name="Mass"),
        Dimension(_idx=2, _sym="L", _unit="m", _name="Length")
    ]
    schema = Schema(_fwk="CUSTOM", _fdu_lt=custom_fdus)

    # Define variables (only 4 relevant for Reynolds number)
    variables = {
        "\\rho": Variable(_name="Density",
                          _sym="\\rho",
                          _cat="IN",
                          _dims="M*L^-3",
                          _units="kg/m³", 
                          relevant=True,
                          _schema=schema),
        "v": Variable(_name="Velocity",
                      _sym="v",
                      _cat="OUT",
                      _dims="L*T^-1",
                      _units="m/s",
                      relevant=True,
                      _schema=schema),
        "D": Variable(_name="Diameter",
                      _sym="D",
                      _cat="IN",
                      _dims="L",
                      _units="m",
                      relevant=True,
                      _schema=schema),
        "\\mu": Variable(_name="Viscosity",
                         _sym="\\mu",
                         _cat="IN",
                         _dims="M*L^-1*T^-1",
                         _units="Pa·s",
                         relevant=True,
                         _schema=schema),
        "g": Variable(_name="Gravity",
                      _sym="g",
                      _cat="CTRL",
                      _dims="L*T^-2",
                      _units="m/s²",
                      relevant=False,
                      _schema=schema)  # Irrelevant for Reynolds
    }

    # Create analysis engine and run workflow
    engine = AnalysisEngine(_name="Reynolds Number Analysis",
                            _fwk="CUSTOM",
                            _schema=schema,
                            _variables=variables)

    # Execute dimensional analysis workflow
    engine.create_matrix()  # Constructs dimensional matrix
    coefficients = engine.solve()  # Generates π coefficients

    # Access Reynolds number coefficient
    reynolds = coefficients["\\Pi_{0}"]
    print(f"Reynolds Formula: {reynolds.pi_expr}")
    print(f"Variables: {list(reynolds.variables.keys())}")

    # Export complete analysis to dictionary (now properly serializes nested objects)
    analysis_data = engine.to_dict()

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Save to JSON file for integration with other systems
    with open("data/reynolds_da.json", "w") as f:
        json.dump(analysis_data, f, indent=4)

    print("Analysis exported to data/reynolds_da.json")

    # Later: Reload analysis from JSON
    with open("data/reynolds_da.json", "r") as f:
        loaded_data = json.load(f)

    # Reconstruct engine from dictionary
    engine_restored = AnalysisEngine.from_dict(loaded_data)
    print(f"Restored coefficients: {list(engine_restored.coefficients.keys())}")

Displayed Capabilities
------------------------

In the example we appreciate the following **PyDASA** capabilities:

1. **Restricts Framework**: Limits dimensional domain to *T*, *M*, *L* only, excluding *A*, *K*, *mol*, *cd* from full *PHYSICAL* framework.
2. **Filters Variables**: Excludes ``relevant=False`` variables (gravity ``g``) from dimensional matrix while retaining them in the model.
3. **Encapsulates Operations**: Handles matrix operations internally without requiring direct ``Matrix`` class access.
4. **Exports State**: Serializes complete analysis state (variables, schema, coefficients) to dictionary for JSON export.
5. **Restores Analysis**: Reconstructs complete ``AnalysisEngine`` from dictionary for reproducibility.

This serialization ensures dimensional analysis results integrate seamlessly with data pipelines, web services, and file-based workflows.