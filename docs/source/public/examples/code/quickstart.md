# Quickstart

Lets try to find the Reynolds number Re = (ρ·v·L)/μ using dimensional analysis.

---

## Step 0: Import PyDASA Dimensional Analysis Module

```python
from pydasa.workflows.phenomena import AnalysisEngine
```

There are two other main modules for Sensitivity Analysis (`SensitivityAnalysis`) and Monte Carlo Simulation (`MonteCarloSimulation`), but we will focus on Dimensional Analysis here.


## Step 1: Define Variables

Define the variables involved in the phenomenon as a dictionary. Each variable is defined by its unique symbolic name (key) and a dictionary of attributes (value).

```python

variables = {
    "\\rho": {
        "_idx": 0,
        "_sym": "\\rho",
        "_fwk": "PHYSICAL",
        "_cat": "IN",              # Category: INPUT variable
        "relevant": True,          # REQUIRED: Include in analysis
        "_dims": "M*L^-3",         # Dimensions: Mass/(Length^3)
        "_setpoint": 1000.0,       # Value for calculations
        "_std_setpoint": 1000.0,   # Standardized value (used internally)
    },
    "v": {
        "_idx": 1,
        "_sym": "v",
        "_fwk": "PHYSICAL",
        "_cat": "OUT",             # OUTPUT - exactly ONE required
        "relevant": True,
        "_dims": "L*T^-1",         # Dimensions: Length/Time
        "_setpoint": 2.0,
        "_std_setpoint": 2.0,
    },
    "L": {
        "_idx": 2,
        "_sym": "L",
        "_fwk": "PHYSICAL",
        "_cat": "IN",
        "relevant": True,
        "_dims": "L",              # Dimensions: Length
        "_setpoint": 0.05,
        "_std_setpoint": 0.05,
    },
    "\\mu": {
        "_idx": 3,
        "_sym": "\\mu",
        "_fwk": "PHYSICAL",
        "_cat": "IN",
        "relevant": True,
        "_dims": "M*L^-1*T^-1",    # Dimensions: Mass/(Length·Time)
        "_setpoint": 0.001,
        "_std_setpoint": 0.001,
    }
}
```

**Notes**:
- Variables with `"relevant": False` are ignored in analysis, even if defined.
- The dimensional matrix needs to have exactly **ONE** output variable (`"_cat": "OUT"`).
- The other bariables can be categoried as Inputs (`"IN"`) or Control (`"CTRL"`).
- `_dims` are the dimensional representations using the current FDUs (Fundamental Dimensional Units) of the selected framework. In this case, we use the `PHYSICAL` framework with base dimensions **M** (Mass), **L** (Length), **T** (Time), but other frameworks are available.
- Subsequent calculations of coefficients require `_setpoint` and `_std_setpoint` values.

---

## Step 2: Create Analysis Engine

To complete the setup, create an `AnalysisEngine` object, specifying the framework and passing the variable definitions. Alternatively, you can add variables later.

```python
engine = AnalysisEngine(_idx=0, _fwk="PHYSICAL")
engine.variables = variables
```

**Notes**:
- By default, the framework is `PHYSICAL`.
- Other built-in frameworks are: `COMPUTATION`, `SOFTWARE`. Plus, you can define custom frameworks with the `CUSTOM` option and a FDU definition list.
- Variables can be added as native dictionaries or as `Variable` **PyDASA** objects (use: `from pydasa.elements.parameter import Variable`).

---

## Step 3: Run Analysis

Then you just run the analysis to solve the dimensional matrix.

```python
results = engine.run_analysis()  # May fail if variable definitions have errors
```

The `run_analysis()` method will process the variables, build the dimensional matrix, and compute the dimensionless coefficients using the Buckingham Pi theorem.

**Output**:

```
Number of dimensionless groups: 1
\Pi_{0}: \mu/(L*\rho*v)
```

**If errors occur**: Check variable definitions (dimensions, categories, relevance flags)

**Notes**:
- The results are stored in `engine.coefficients` as `Coefficient` objects.
- Each coefficient has attributes like `pi_expr` (the dimensionless expression), `name`, `symbol`, etc. used for further analysis, visualization, or exporting.
- The variables are accessible via `engine.variables` for any additional processing or exporting.

---

## Step 4: Display Results

Then, you can display the results in the console or export them for visualization.

Here is how you print the coefficients:

```python
for name, coeff in engine.coefficients.items():
    print(f"{name}: {coeff.pi_expr}")
```

If you want to export the results for use with external libraries (matplotlib, pandas, seaborn) you can use the `to_dict()` method:

```python
# Export to dict for external libraries
data_dict = list(engine.coefficients.values())[0].to_dict()

# Example: Use with pandas
import pandas as pd
df = pd.DataFrame([data_dict])

# Example: Access variables for plotting
var_data = {sym: var.to_dict() for sym, var in engine.variables.items()}
```

---

## Step 5: Derive \& Calculate Coefficients

You can also derive new coefficients from existing ones and calculate their numerical values using the stored setpoints.

```python
# Derive Reynolds number (Re = 1/Pi_0)
pi_0_key = list(engine.coefficients.keys())[0]
Re_coeff = engine.derive_coefficient(
    expr=f"1/{pi_0_key}",
    symbol="Re",
    name="Reynolds Number"
)

# Calculate numerical value using stored setpoints
Re_value = Re_coeff.calculate_setpoint()  # Uses _std_setpoint values
print(f"Reynolds Number: {Re_value:.2e}")
```

**Notes**:
- The `derive_coefficient()` method allows you to create new coefficients based on existing ones using mathematical expressions.
- The `calculate_setpoint()` method computes the numerical value of the coefficient using the `_std_setpoint` values of the involved variables.
- The other **PyDASA** modules (Sensitivity Analysis, Monte Carlo Simulation) also use the `Variable` and `Coefficient` objects, so you can seamlessly integrate dimensional analysis results into further analyses.

**Output**:

```
Reynolds Number: 1.00e+05
Flow regime: TURBULENT
```

---

## Summary

| Step | Action              | Notes                                                                    |
| ---- | ------------------- | ----------------------------------------------------------------------------- |
| 1    | Define variables    | important attributes `relevant=True`, exactly 1 `_cat=OUT`, try to include `_setpoint`/`_std_setpoint`. |
| 2    | Create engine       | `_fwk="PHYSICAL"` (or custom), accepts `dict` or `Variable` objects.            |
| 3    | Run analysis        | `run_analysis()` may fail on ill defined variables, inconsistent units, missing attributes, or invalid FDUs.                                         |
| 4    | Display results     | Console output or export via `.to_dict()` to use other libraries.                                   |
| 5    | Derive coefficients | Use `derive_coefficient()` + `calculate_setpoint()` to compute new coefficients and their values.                      |

**Full example**: See `reynolds_simple.py` in the PyDASA repository.

**Explore more**: Visit the [PyDASA Documentation](https://pydasa.readthedocs.io) for advanced features, tutorials, and API reference.
