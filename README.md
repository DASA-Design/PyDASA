# PyDASA

![PyPI](https://img.shields.io/pypi/v/pydasa)
![Python Version](https://img.shields.io/pypi/pyversions/pydasa)
![License](https://img.shields.io/github/license/DASA-Design/PyDASA)
![Documentation Status](https://readthedocs.org/projects/pydasa/badge/?version=latest)
![Coverage](https://codecov.io/gh/DASA-Design/PyDASA/branch/main/graph/badge.svg)

**PyDASA** (Dimensional Analysis for Scientific Applications and Software Architecture) is an open-source Python library for dimensional analysis of complex phenomena across physical, chemical, computational, and software domains using the Buckingham Pi-theorem.

## The Need (Epic User Story)

**As a** researcher, engineer, or software architect analyzing complex systems,

**I want** a comprehensive dimensional analysis library implementing the Buckingham Pi theorem,

**So that** I can systematically discover dimensionless relationships, validate models, and understand system behavior across physical, computational, and software architecture domains.

## Installation

To install **PyDASA**, use pip:

```bash
pip install pydasa
```

Then, to check the installed version of **PyDASA**, run:

```python
import pydasa
print(pydasa.__version__)
```

## Quickstart

Lets try to find the Reynolds number Re = (ρ·v·L)/μ using dimensional analysis.

---

### Step 0: Import PyDASA Dimensional Analysis Module

```python
from pydasa.workflows.phenomena import AnalysisEngine
```

There are two other main modules for Sensitivity Analysis (`SensitivityAnalysis`) and Monte Carlo Simulation (`MonteCarloSimulation`), but we will focus on Dimensional Analysis here.

### Step 1: Define Variables

Define the variables involved in the phenomenon as a dictionary. Each variable is defined by its unique symbolic name (key) and a dictionary of attributes (value).

```python

# Define variables for Reynolds number example
# Can be a list, dict, or Variable objects
variables = {
    # Density: ρ [M/L³] - INPUT
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
    # Velocity in pipe: v [L/T] - OUTPUT (we want to find this)
    "v": {
        "_idx": 1,
        "_sym": "v",
        "_fwk": "PHYSICAL",
        "_cat": "IN",           # if this were OUT, Reynolds would be trivial
        "relevant": True,
        "_dims": "L*T^-1",      # Dimensions: Length/Time
        "_setpoint": 5.0,
        "_std_setpoint": 5.0,
    },
    "D": {      # pipe diameter
        "_idx": 2,
        "_sym": "D",
        "_fwk": "PHYSICAL",
        "_cat": "IN",
        "relevant": True,
        "_dims": "L",               # Dimensions: Length
        "_setpoint": 0.05,
        "_std_setpoint": 0.05,
    },
    # Length: L [L] - INPUT
    "\\mu": {
        "_idx": 3,
        "_sym": "\\mu",
        "_fwk": "PHYSICAL",
        "_cat": "OUT",              # Need exactly one OUTPUT variable
        "relevant": True,
        "_dims": "M*L^-1*T^-1",     # Dimensions: Mass/(Length·Time)
        "_setpoint": 0.001,
        "_std_setpoint": 0.001,
    }
}
```

**Notes**:

- Variables with `"relevant": False` are ignored in analysis, even if defined.
- The dimensional matrix needs to have exactly **ONE** output variable (`"_cat": "OUT"`).
- The other variables can be categorized as Inputs (`"IN"`) or Control (`"CTRL"`).
- `_dims` are the dimensional representations using the current FDUs (Fundamental Dimensional Units) of the selected framework. In this case, we use the `PHYSICAL` framework with base dimensions **M** (Mass), **L** (Length), **T** (Time), but other frameworks are available.
- Subsequent calculations of coefficients require `_setpoint` and `_std_setpoint` values.

---

### Step 2: Create Analysis Engine

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

### Step 3: Run Analysis

Then you just run the analysis to solve the dimensional matrix.

```python
results = engine.run_analysis()  # May fail if variable definitions have errors
print(f"Number of dimensionless groups: {len(results)}")
for name, coeff in results.items():
    print(f"\t{name}: {coeff.get('pi_expr')}")
```

The `run_analysis()` method will process the variables, build the dimensional matrix, and compute the dimensionless coefficients using the Buckingham Pi theorem; printing and processing the results in dict format will show the number of dimensionless groups found and their expressions.

**Output**:

```
Number of dimensionless groups: 1
    \Pi_{0}: \frac{\mu}{\rho*v*L}
```

**If errors occur**: Check variable definitions (dimensions, categories, relevance flags)

**Notes**:

- The results are stored in `engine.coefficients` as `Coefficient` objects.
- Each coefficient has attributes like `pi_expr` (the dimensionless expression), `name`, `symbol`, etc. used for further analysis, visualization, or exporting.
- The variables are accessible via `engine.variables` for any additional processing or exporting.

---

### Step 4: Display Results

Then, you can also display the object-like results in console or export them for visualization.

Here is how you print the coefficients:

```python
print(f"Number of dimensionless groups: {len(engine.coefficients)}")
for name, coeff in engine.coefficients.items():
    print(f"\t{name}: {coeff.pi_expr}")
    print(f"\tVariables: {list(coeff.var_dims.keys())}")
    print(f"\tExponents: {list(coeff.var_dims.values())}")
```

Then, the output will be:

```
Number of dimensionless groups: 1
        \Pi_{0}: \frac{\mu}{\rho*v*L}
        Variables: ['\\rho', 'v', 'L', '\\mu']
        Exponents: [-1, -1, -1, 1]
```

Since variables and coefficients are Python objects, you can export them to dict format for external libraries (matplotlib, pandas, seaborn) using `to_dict()`:

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

### Step 5: Derive \& Calculate Coefficients

Since expressions and setpoints are stored in variables, you can derive new coefficients from existing ones and calculate their values directly.

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

# Interpret the result based on typical flow regimes
if Re_value < 2300:
    print("Flow regime: LAMINAR")
elif Re_value < 4000:
    print("Flow regime: TRANSITIONAL")
else:
    print("Flow regime: TURBULENT")
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

### Summary

| Step | Action              | Notes                                                                                                           |
| ---- | ------------------- | --------------------------------------------------------------------------------------------------------------- |
| 1    | Define variables    | important attributes `relevant=True`, exactly 1 `_cat=OUT`, try to include `_setpoint`/`_std_setpoint`. |
| 2    | Create engine       | `_fwk="PHYSICAL"` (or custom), accepts `dict` or `Variable` objects.                                      |
| 3    | Run analysis        | `run_analysis()` may fail on ill defined variables, inconsistent units, missing attributes, or invalid FDUs.  |
| 4    | Display results     | Console output or export via `.to_dict()` to use other libraries.                                             |
| 5    | Derive coefficients | Use `derive_coefficient()` + `calculate_setpoint()` to compute new coefficients and their values.           |

**Full example**: See `reynolds_simple.py` in the PyDASA repository.

**Explore more**: Visit the [PyDASA Documentation](https://pydasa.readthedocs.io) for advanced features, tutorials, and API reference.

## Core Capabilities

### Manage Dimensional Domain

- **Manage Fundamental Dimensions** beyond traditional physical units (L, M, T) .to include computational (T, S, N) and software architecture domains (T, D, E, C, A).
- **Switch between frameworks** for different problem domains.

### Manage Symbolic and Numerical Variables

- **Define dimensional parameters** with complete specifications:
  - **Specify** symbolic representation (name, LaTeX symbol).
  - **Define** dimensional formula (e.g., "L*T^-1" for velocity).
  - **Establish** numerical ranges (min, max, mean, step)
  - **Assign** classification (input, output, control).
  - **Configure** statistical distributions and dependencies.

### Integrate System of Units of Measurement

- **Handle measurements** across unit systems (imperial, metric, custom).
- **Convert between units** while maintaining dimensional consistency.
- **Relate measurements** to dimensional parameters.

### Discover Dimensionless Coefficients

- **Generate dimensionless numbers** using the Buckingham Pi theorem:
  1. **Build relevance list** by identifying mutually independent parameters influencing the phenomenon.
  2. **Construct dimensional matrix** by arranging FDUs (rows) and variables (columns) into core and residual matrices.
  3. **Transform to identity matrix** by applying linear transformations to the core matrix.
  4. **Generate Pi coefficients** by combining residual and unity matrices to produce dimensionless groups.
- **Classify coefficients** by repeating vs. non-repeating parameters.
- **Manage metadata:** names, symbols, formulas, and parameter relationships.

### Analyze and Simulate Coefficient Behavior

- **Verify similitude principles** for model scaling and validation.
- **Calculate coefficient ranges** and parameter influence.
- **Run Monte Carlo simulations** to quantify uncertainty propagation.
- **Perform sensitivity analysis** to identify dominant parameters.
- **Generate behavioral data** for dimensionless relationships.

### Export, Integrate, and Visualize Data

- **Export data formats** compatible with pandas, matplotlib, seaborn.
- **Structure results** for integration with visualization libraries.
- **Provide standardized outputs** for dimensionless charts and parameter influence plots.

## Documentation

For more information on how to se **PyDASA**, go to our comprehensive documentation pagee available at [readthedocs.io](https://pydasa.readthedocs.io/en/latest/).

### Development Status

**Emoji Convention:**
    - 📋 TODO
    - 🔶👨‍💻 WORKING
    - ✅ DONE
    - ⚠️ ATTENTION REQUIRED

**Current Version:** 0.6.4

### ✅ Core Modules (Implemented & Tested)

- **core/**: Foundation classes, configuration, I/O.
- **dimensional/**: Buckingham Pi theorem, dimensional matrix solver.
- **elements/**: Variable and parameter management with specs.
- **workflows/**: AnalysisEngine, MonteCarloSimulation, SensitivityAnalysis.
- **validations/**: Decorator-based validation system.
- **serialization/**: LaTeX and formula parsing.

### 👨‍💻 Currently Working

- **Documentation**: Improving API reference, tutorials, and user guides.
- **Code Reduction**: Refactoring to eliminate redundancy, improve maintainability, readability, and performance.
- **Data Structures**: Designing implementation for unit of measure and dimensional management systems to enable consistent unit conversion across frameworks.

### 📋 Pending Development

- **context/**: Implement Unit conversion system (stub implementation).
- **structs/**: Implement Data structures (partial test coverage).
- **Documentation**: Complete API reference completion and additional tutorials.

## ⚠️ How to Contribute

Contributions are welcome! We use [Conventional Commits](https://www.conventionalcommits.org/) for automatic versioning and changelog generation.

### Commit Message Format

```
<type>(<scope>): <subject>
```

**Types:**

- `feat`: New feature (triggers MINOR version bump).
- `fix`: Bug fix (triggers PATCH version bump).
- `docs`: Documentation changes only.
- `refactor`: Code refactoring without feature changes..
- `test`: Adding or updating tests.
- `perf`: Performance improvements.
- `chore`: Other changes that don't modify src or test files.

**Breaking Changes:** Add `BREAKING CHANGE:` in commit footer to trigger MAJOR version bump.

### Examples

```bash
# Feature (0.6.0 → 0.7.0)
git commit -m "feat(workflows): add uncertainty propagation analysis"

# Bug fix (0.6.0 → 0.6.1)
git commit -m "fix(buckingham): resolve matrix singularity edge case"

# Breaking change (0.6.0 → 1.0.0)
git commit -m "feat(api)!: redesign Variable API

BREAKING CHANGE: Variable.value renamed to Variable.magnitude"
```

### Development Workflow

```bash
# Clone and setup
git clone https://github.com/DASA-Design/PyDASA.git
cd PyDASA

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Commit with conventional format
git commit -m "feat(module): add new feature"

# Create PR for review
```

### Release Process

1. Make changes with conventional commit messages.
2. Create PR and merge to `main`.
3. GitHub Actions automatically:
   - Analyzes commit messages.
   - Bumps version (MAJOR.MINOR.PATCH)..
   - Updates `_version.py` and `pyproject.toml`.
   - Creates GitHub release with changelog.
   - Publishes to PyPI.

For more details, visit our [Contributing Guide](https://pydasa.readthedocs.io/en/latest/development/contributing.html).
