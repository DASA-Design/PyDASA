# PyDASA: Code Documentation Report
**Version:** 0.7.0 | **Licence:** GPL-3.0
**Repository:** https://github.com/DASA-Design/PyDASA
**Style:** Google docstring style (Sphinx-compatible)

---

## Overview

PyDASA (*Python Dimensional Analysis for Software Architecture*) is a domain-agnostic Python library for applying the Buckingham Pi theorem to dimensional analysis problems across physical, computational, and software architecture domains. The library exposes a layered public API spanning six operable modules. This report documents the complete public API of each module, including all public classes, methods, and their parameters, using Google-style docstrings throughout.

---

## Module: `pydasa.workflows.phenomena`

The workflows module is the primary user-facing entry point. It provides three composable workflow classes: `AnalysisEngine` for dimensional analysis, `SensitivityAnalysis` for parameter perturbation studies, and `MonteCarloSimulation` for uncertainty propagation. All three consume `Variable` objects and produce `Coefficient` objects, making them directly composable without intermediate conversion.

**Typical usage:**

```python
from pydasa.workflows.phenomena import AnalysisEngine

engine = AnalysisEngine(_idx=0, _fwk="PHYSICAL")
engine.variables = variables_dict
results = engine.run_analysis()
```

---

### Class: `AnalysisEngine`

```python
class AnalysisEngine:
    """Orchestrates the complete dimensional analysis pipeline.

    Accepts a set of dimensional variables, validates them, constructs the
    dimensional matrix, applies the Buckingham Pi theorem, and stores the
    resulting dimensionless Π coefficients. This is the primary entry point
    for all dimensional analysis workflows in PyDASA.

    The engine accepts variables either as plain Python dicts or as typed
    Variable objects. Both forms are normalised transparently before
    processing. Exactly one variable must have _cat='OUT'; all others
    must be 'IN' or 'CTRL'. At least one variable must have relevant=True.

    Attributes:
        _idx (int): Unique integer identifier for this engine instance.
        _fwk (str): The selected dimensional schema identifier. One of
            'PHYSICAL', 'COMPUTATION', 'SOFTWARE', or 'CUSTOM'.
        _fdu_list (list[str] | None): Required when _fwk='CUSTOM'. A list
            of FDU symbol strings defining the custom schema.
        variables (dict): The variable dictionary, keyed by symbolic name.
            Setting this property triggers full validation of the variable
            set. Values may be plain dicts or Variable objects.
        coefficients (dict): The Π coefficient dictionary, populated after
            run_analysis() is called. Keys are Pi group labels (e.g.,
            'Pi_0'); values are Coefficient objects.

    Example:
        >>> engine = AnalysisEngine(_idx=0, _fwk="PHYSICAL")
        >>> engine.variables = {"rho": {...}, "v": {...}, "L": {...}, "mu": {...}}
        >>> results = engine.run_analysis()
        >>> print(results["Pi_0"].pi_expr)
        mu/(L*rho*v)
    """
```

#### Method: `__init__`

```python
def __init__(self, _idx: int, _fwk: str, _fdu_list: list[str] | None = None):
    """Initialise the AnalysisEngine with a dimensional schema.

    Args:
        _idx: Unique integer identifier for this engine instance. Used
            to distinguish multiple engines in a multi-schema analysis.
        _fwk: The dimensional schema to use. Built-in options are
            'PHYSICAL' (FDUs: M, L, T), 'COMPUTATION' (FDUs: T, S, N),
            and 'SOFTWARE' (FDUs: T, D, E, C, A). Use 'CUSTOM' with
            _fdu_list to define an arbitrary schema.
        _fdu_list: Required when _fwk='CUSTOM'. A list of FDU symbol
            strings, e.g. ['M', 'T', 'C'] for a biological schema.
            Must contain at least one symbol. Symbols must be unique.

    Raises:
        ValueError: If _fwk='CUSTOM' and _fdu_list is None or empty.
        ValueError: If _fwk is not one of the recognised schema identifiers.
    """
```

#### Method: `run_analysis`

```python
def run_analysis(self) -> dict:
    """Execute the full dimensional analysis pipeline.

    Validates the variable set, constructs the dimensional matrix,
    reduces the core sub-matrix to an identity matrix via linear row
    operations, and produces the n - k dimensionless Π groups. Results
    are stored in self.coefficients and also returned.

    The number of Π groups produced is always n - k, where n is the
    count of relevant variables and k is the rank of the dimensional
    matrix (equal to the number of FDUs in a well-posed problem).

    Returns:
        dict: The populated coefficients dictionary, mapping Pi group
            labels (e.g., 'Pi_0', 'Pi_1') to Coefficient objects.

    Raises:
        ValidationError: If no variables have been assigned, if there
            is not exactly one output variable, or if any variable has
            an unparseable dimensional formula.
        SingularMatrixError: If the core sub-matrix is singular,
            indicating that the chosen repeating variables are not
            dimensionally independent. The exception message names the
            offending variable.

    Example:
        >>> results = engine.run_analysis()
        Number of dimensionless groups: 1
        Pi_0: mu/(L*rho*v)
    """
```

#### Method: `derive_coefficient`

```python
def derive_coefficient(
    self,
    expr: str,
    symbol: str,
    name: str,
) -> "Coefficient":
    """Derive a new named dimensionless coefficient from existing primary groups.

    Creates a Coefficient by evaluating a symbolic expression over the
    engine's existing coefficients. Useful for recovering well-known
    dimensionless numbers (e.g., the Reynolds number as the reciprocal
    of the primary Π group).

    Args:
        expr: Symbolic expression string using existing coefficient
            labels as variable names. For example, '1/Pi_0' or
            'Pi_0 * Pi_1'. Standard Python arithmetic operators are
            supported. Coefficient labels must match keys in
            self.coefficients exactly.
        symbol: Short symbolic identifier for the new coefficient,
            e.g. 'Re'. Used in LaTeX rendering and export metadata.
        name: Human-readable name, e.g. 'Reynolds Number'. Used in
            DataFrame column headers and report generation.

    Returns:
        Coefficient: A new Coefficient object with pi_expr set to the
            evaluated symbolic expression and calculate_setpoint()
            bound to the constituent variables' _std_setpoint values.

    Raises:
        KeyError: If expr references a coefficient label that does not
            exist in self.coefficients.
        ValueError: If expr is not a valid symbolic expression.
        RuntimeError: If run_analysis() has not been called before
            this method is invoked.

    Example:
        >>> pi_0_key = list(engine.coefficients.keys())[0]
        >>> Re = engine.derive_coefficient(
        ...     expr=f"1/{pi_0_key}",
        ...     symbol="Re",
        ...     name="Reynolds Number",
        ... )
        >>> print(Re.calculate_setpoint())
        100000.0
    """
```

---

### Class: `SensitivityAnalysis`

```python
class SensitivityAnalysis:
    """Evaluates how perturbations in variable setpoints propagate to Π values.

    Accepts a set of Coefficient objects produced by AnalysisEngine and
    systematically perturbs each variable's _std_setpoint by a configurable
    factor. For each perturbation, the resulting change in each Π coefficient
    value is measured. This identifies the variables that most strongly govern
    each dimensionless relationship and quantifies the robustness of the
    analysis to measurement uncertainty.

    Coefficient objects from AnalysisEngine are accepted directly without
    conversion or wrapping.

    Attributes:
        coefficients (dict): Coefficient objects from AnalysisEngine.
        perturbation_factor (float): Fractional perturbation applied to
            each variable setpoint. Default is 0.01 (1%).

    Example:
        >>> sa = SensitivityAnalysis(coefficients=engine.coefficients)
        >>> report = sa.run()
        >>> print(report)  # dict mapping variable symbols to sensitivity scores
    """
```

#### Method: `run`

```python
def run(self) -> dict:
    """Execute the sensitivity analysis over all constituent variables.

    For each variable in the coefficient's variable set, perturbs
    _std_setpoint by ±perturbation_factor and measures the resulting
    change in each Π coefficient value. Returns a structured report.

    Returns:
        dict: Sensitivity report mapping variable symbolic names to a
            dict of Π group labels and their sensitivity scores. A
            score of 1.0 indicates a 1:1 proportional relationship
            between variable perturbation and coefficient change.

    Raises:
        ValueError: If coefficients is empty or contains no variables
            with _std_setpoint values set.

    Example:
        >>> report = sa.run()
        >>> # {"rho": {"Pi_0": 1.0}, "v": {"Pi_0": 1.0}, ...}
    """
```

---

### Class: `MonteCarloSimulation`

```python
class MonteCarloSimulation:
    """Propagates statistical uncertainty through the dimensional model.

    Accepts Coefficient objects from AnalysisEngine and samples variable
    distributions n_samples times. For each sample, the Π coefficient
    values are computed from the sampled setpoints. The resulting
    distributions of Π values yield confidence intervals for each
    dimensionless relationship.

    Supports two input modes for variable values: (1) statistical
    distribution parameters defined on Variable objects (mean, std,
    distribution type), and (2) pre-existing empirical datasets supplied
    as lists of observed values, which is particularly useful for
    validation against experimental measurements.

    Attributes:
        coefficients (dict): Coefficient objects from AnalysisEngine.
        n_samples (int): Number of Monte Carlo samples. Default is 1000.
        empirical_data (dict | None): Optional dict mapping variable
            symbolic names to lists of observed float values. When
            provided for a variable, empirical sampling replaces
            distribution-based sampling for that variable.
        confidence_level (float): Confidence interval level, between
            0.0 and 1.0. Default is 0.95 (95% CI).

    Example:
        >>> mc = MonteCarloSimulation(
        ...     coefficients=engine.coefficients,
        ...     n_samples=10000,
        ... )
        >>> result = mc.run()
        >>> print(result["Pi_0"]["ci_95"])  # (lower_bound, upper_bound)
    """
```

#### Method: `run`

```python
def run(self) -> dict:
    """Execute the Monte Carlo simulation.

    Samples each variable n_samples times from its distribution
    parameters or empirical dataset, computes the resulting Π coefficient
    values for each sample, and returns the distribution summary.

    Returns:
        dict: Result mapping each Π group label to a dict containing:
            - 'mean' (float): Mean Π value across all samples.
            - 'std' (float): Standard deviation of Π values.
            - 'ci_{level*100}' (tuple[float, float]): Confidence interval
              as (lower, upper) bounds at the configured confidence level.
            - 'samples' (list[float]): Raw sample values if store_samples
              is True (default False to conserve memory).

    Raises:
        ValueError: If n_samples < 2.
        ValueError: If a variable has neither distribution parameters
            nor an empirical_data entry.
        NumericalWarning: If one or more samples produce a numerical
            overflow. Failed samples are logged, skipped, and excluded
            from the distribution summary.

    Example:
        >>> result = mc.run()
        >>> lower, upper = result["Pi_0"]["ci_95"]
        >>> print(f"95% CI: ({lower:.4f}, {upper:.4f})")
    """
```

---

## Module: `pydasa.elements.parameter`

The elements module defines the two domain objects that flow between all layers of the library: `Variable` and `Coefficient`. These objects constitute the shared data contract. Both expose `to_dict()` as the primary interoperability hook.

---

### Class: `Variable`

```python
class Variable:
    """Encapsulates a single dimensional parameter.

    A Variable is the fundamental unit of data in PyDASA. It carries the
    complete specification of a dimensional quantity: its symbolic identity,
    dimensional formula expressed in the selected schema's FDUs, numerical
    setpoints for evaluation, category within the analysis, and optional
    statistical distribution parameters for simulation workflows.

    Variables may be created from plain Python dicts or constructed directly.
    AnalysisEngine normalises both forms transparently.

    Attributes:
        _idx (int): Unique integer index within the variable set.
        _sym (str): Symbolic name, typically a LaTeX string (e.g., '\\rho').
            Used as the dictionary key and in Pi expression rendering.
        _fwk (str): Schema identifier ('PHYSICAL', 'COMPUTATION',
            'SOFTWARE', or 'CUSTOM').
        _cat (str): Variable category. One of 'IN' (input), 'OUT' (output,
            exactly one required per analysis), or 'CTRL' (control variable,
            held constant during simulation).
        relevant (bool): Analysis inclusion flag. Variables with
            relevant=False are excluded from the dimensional matrix.
        _dims (str): Dimensional formula string using the schema's FDU
            symbols and standard operators. Examples: 'M*L^-3',
            'L*T^-1', 'M*L^-1*T^-1'. Must be parseable by the
            serialization module.
        _setpoint (float): Nominal value of the variable in its natural
            units. Used for display and initial evaluation.
        _std_setpoint (float): Standardised value used internally for
            coefficient numerical evaluation. Typically equal to
            _setpoint but may differ after unit normalisation.
        distribution_params (dict | None): Optional statistical
            distribution specification for Monte Carlo simulation.
            Keys: 'type' (str, e.g. 'normal', 'uniform', 'lognormal'),
            'mean' (float), 'std' (float), 'min' (float), 'max' (float).

    Example:
        >>> var = Variable({
        ...     "_idx": 0, "_sym": "\\\\rho", "_fwk": "PHYSICAL",
        ...     "_cat": "IN", "relevant": True,
        ...     "_dims": "M*L^-3",
        ...     "_setpoint": 1000.0, "_std_setpoint": 1000.0,
        ... })
        >>> print(var._dims)
        M*L^-3
    """
```

#### Method: `to_dict`

```python
def to_dict(self) -> dict:
    """Export all variable attributes as a plain Python dictionary.

    Produces a flat dictionary compatible with pandas DataFrame
    construction, matplotlib plotting, and JSON serialisation without
    any custom adapter. The output dictionary contains all public
    attributes of the Variable object.

    Returns:
        dict: A flat dictionary with string keys corresponding to
            Variable attribute names and their current values.
            All values are Python primitives (str, int, float, bool,
            dict, or None) suitable for direct use with pandas.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame([var.to_dict()])
        >>> print(df.columns.tolist())
        ['_idx', '_sym', '_fwk', '_cat', 'relevant', '_dims',
         '_setpoint', '_std_setpoint', 'distribution_params']
    """
```

---

### Class: `Coefficient`

```python
class Coefficient:
    """Encapsulates a dimensionless Π group derived from the Buckingham Pi theorem.

    A Coefficient represents one dimensionless relationship produced by the
    Pi solver, or a named group derived from primary Π groups via algebraic
    combination. Each Coefficient carries its symbolic expression, derivation
    metadata, and a reference to the constituent Variable objects required
    for numerical evaluation.

    Attributes:
        pi_expr (str): The symbolic dimensionless expression as a string,
            e.g. 'mu/(L*rho*v)'. Rendered to LaTeX by the serialization
            module for documentation output.
        symbol (str): Short identifier for the coefficient, e.g. 'Pi_0'
            for primary groups or 'Re' for derived named numbers.
        name (str): Human-readable name, e.g. 'Reynolds Number'.
        variables (list[Variable]): The constituent Variable objects whose
            _std_setpoint values are used in calculate_setpoint().

    Example:
        >>> coeff = engine.coefficients["Pi_0"]
        >>> print(coeff.pi_expr)
        mu/(L*rho*v)
        >>> print(coeff.calculate_setpoint())
        1e-05
    """
```

#### Method: `calculate_setpoint`

```python
def calculate_setpoint(self) -> float:
    """Evaluate the coefficient numerically from constituent variable setpoints.

    Substitutes the _std_setpoint value of each constituent Variable into
    the symbolic pi_expr and evaluates the result numerically. This
    provides the numerical value of the dimensionless group at the
    operating point defined by the variable setpoints.

    Returns:
        float: The numerical value of the coefficient at the current
            variable setpoints.

    Raises:
        ValueError: If any constituent variable has _std_setpoint=None.
        ZeroDivisionError: If the evaluated expression has a zero
            denominator at the given setpoints.

    Example:
        >>> # For Reynolds number with rho=1000, v=2, L=0.05, mu=0.001:
        >>> Re = engine.derive_coefficient(expr="1/Pi_0", symbol="Re",
        ...                               name="Reynolds Number")
        >>> print(Re.calculate_setpoint())
        100000.0
    """
```

#### Method: `to_dict`

```python
def to_dict(self) -> dict:
    """Export all coefficient attributes as a plain Python dictionary.

    Produces a flat dictionary compatible with pandas DataFrame
    construction and JSON serialisation. Constituent Variable objects
    are included as nested dicts via their own to_dict() method.

    Returns:
        dict: A flat dictionary with keys 'pi_expr', 'symbol', 'name',
            'setpoint_value' (float, result of calculate_setpoint() if
            setpoints are available, else None), and 'variables' (list
            of Variable.to_dict() outputs).

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame([coeff.to_dict()])
        >>> print(df["pi_expr"].iloc[0])
        mu/(L*rho*v)
    """
```

---

## Module: `pydasa.dimensional`

The dimensional module implements the Buckingham Pi theorem solver. It is the mathematical core of the library. This module is independently testable and does not depend on the workflows layer.

**Module docstring:**

```python
"""
pydasa.dimensional — Buckingham Pi theorem solver.

Implements the four-step dimensional analysis algorithm: relevance
filtering, dimensional matrix construction, core matrix row reduction,
and Π coefficient generation. Operates on Variable objects from the
elements layer and produces Coefficient objects consumed by the
workflows layer.

This module is the mathematical core of PyDASA and is independently
testable without instantiating AnalysisEngine.

Typical usage (via AnalysisEngine, which delegates to this module)::

    engine = AnalysisEngine(_idx=0, _fwk="PHYSICAL")
    engine.variables = variables
    engine.run_analysis()   # internally calls PiSolver
"""
```

### Key Internal Class: `PiSolver`

```python
class PiSolver:
    """Implements the four-step Buckingham Pi algorithm.

    Operates on a filtered list of Variable objects and produces
    a list of primary Coefficient objects. Called internally by
    AnalysisEngine; not part of the public API but documented here
    for completeness and testability reference.

    Args:
        variables (list[Variable]): Filtered list of relevant Variable
            objects. Must contain at least one variable and exactly one
            with _cat='OUT'.
        fdu_list (list[str]): Ordered list of FDU symbol strings for
            the selected schema.

    Step 1 — Relevance filtering: variables with relevant=False excluded.
    Step 2 — Dimensional matrix construction: FDUs as rows, variables
        as columns. Partitioned into core (repeating variables) and
        residual (non-repeating, including output variable) sub-matrices.
    Step 3 — Core matrix reduction: linear row operations reduce core
        to identity; same operations applied to residual simultaneously.
    Step 4 — Π generation: residual and unity matrices combined to
        yield n - k Coefficient objects.

    Raises:
        SingularMatrixError: If the core sub-matrix is rank-deficient.
    """
```

---

## Module: `pydasa.validations`

The validations module implements a decorator-based constraint system. Decorators are attached to methods throughout the library, enforcing structural constraints at call time and keeping validation concerns orthogonal to business logic.

**Module docstring:**

```python
"""
pydasa.validations — Decorator-based validation system.

Provides decorators that enforce structural constraints on variable
sets and engine state before delegating to the decorated method.
Validation occurs at the earliest possible point: at variable
assignment and engine instantiation, not at analysis runtime.

Key enforced constraints:
- Exactly one variable with _cat='OUT'.
- At least one variable with relevant=True.
- All _dims strings parseable as valid dimensional formulae.
- All FDU symbols referenced in _dims present in the declared schema.

Violations raise ValidationError with a message identifying the
specific variable and attribute that caused the failure.

Each decorator is independently unit-testable without instantiating
AnalysisEngine or the Pi solver.
"""
```

### Key Decorator: `validate_variables`

```python
def validate_variables(func):
    """Decorator enforcing variable set structural constraints before delegation.

    Checks that the variable set assigned to the engine satisfies all
    constraints required for a valid dimensional analysis. Raises
    ValidationError with a targeted message on any violation, identifying
    the specific variable symbol and attribute at fault.

    Constraints enforced:
        - Exactly one variable has _cat='OUT'.
        - At least one variable has relevant=True.
        - All _dims strings are parseable by the serialization module.
        - All FDU symbols in _dims are present in the declared schema.

    Args:
        func: The method to wrap. Expected signature must include self
            (an AnalysisEngine instance with self._fwk and self.variables).

    Returns:
        Callable: Wrapped method that validates before delegation.

    Raises:
        ValidationError: On any constraint violation. Message format:
            "Variable '{symbol}': {constraint} violated. {description}."

    Example:
        @validate_variables
        def run_analysis(self) -> dict:
            ...
    """
```

---

## Module: `pydasa.serialization`

The serialization module handles bidirectional parsing between string dimensional formulae and their internal matrix representations, and renders Pi expressions to LaTeX for documentation output.

**Module docstring:**

```python
"""
pydasa.serialization — Formula parsing and LaTeX rendering.

Provides bidirectional parsing between string dimensional formulae
(e.g., 'M*L^-3') and their internal representation as exponent vectors,
and renders symbolic Pi expressions to LaTeX strings for use in
Sphinx documentation and ReadTheDocs builds.

Key functions:
- parse_dims(dims_str, fdu_list): str -> list[float] exponent vector.
- render_latex(pi_expr): str -> LaTeX-formatted string.
- validate_dims(dims_str, fdu_list): str -> bool.
"""
```

### Key Function: `parse_dims`

```python
def parse_dims(dims_str: str, fdu_list: list[str]) -> list[float]:
    """Parse a dimensional formula string into an exponent vector.

    Converts a string dimensional formula expressed using schema FDU
    symbols and standard operators into a list of exponents, one per
    FDU in the schema, ordered to match fdu_list.

    Args:
        dims_str: Dimensional formula string. Uses FDU symbols from
            fdu_list combined with '*' (multiplication) and '^'
            (exponentiation). Examples: 'M*L^-3', 'L*T^-1', 'M'.
            Exponents may be integer or fractional. Symbols not
            appearing in dims_str are assigned exponent 0.
        fdu_list: Ordered list of FDU symbol strings for the schema,
            e.g. ['M', 'L', 'T'] for PHYSICAL. Determines the length
            and ordering of the output vector.

    Returns:
        list[float]: Exponent vector of length len(fdu_list). The i-th
            element is the exponent of fdu_list[i] in dims_str.
            Example: parse_dims('M*L^-3', ['M', 'L', 'T']) -> [1, -3, 0].

    Raises:
        ValueError: If dims_str contains an unrecognised symbol not in
            fdu_list, or if dims_str is malformed (e.g., empty string,
            invalid operator syntax such as '**').

    Example:
        >>> parse_dims("M*L^-3", ["M", "L", "T"])
        [1.0, -3.0, 0.0]
        >>> parse_dims("L*T^-1", ["M", "L", "T"])
        [0.0, 1.0, -1.0]
    """
```

---

## Module: `pydasa.core`

The core module provides base classes, shared configuration constants, and I/O utilities used across all other modules.

**Module docstring:**

```python
"""
pydasa.core — Foundation classes, configuration, and I/O utilities.

Provides the base classes from which Variable, Coefficient, and
workflow classes inherit, along with shared configuration constants
(built-in schema FDU definitions) and utility functions for
reading and writing analysis state.

This module has no dependencies on any other pydasa module and is
the foundation of the unidirectional dependency structure.

Built-in schema definitions:
    PHYSICAL_FDUS = ['M', 'L', 'T']
    COMPUTATION_FDUS = ['T', 'S', 'N']
    SOFTWARE_FDUS = ['T', 'D', 'E', 'C', 'A']
"""
```

---

## API Quick Reference

| Class / Function | Module | Description |
|-----------------|--------|-------------|
| `AnalysisEngine` | `workflows.phenomena` | Primary entry point; orchestrates full dimensional analysis pipeline |
| `SensitivityAnalysis` | `workflows.phenomena` | Perturbs variable setpoints; measures Π coefficient sensitivity |
| `MonteCarloSimulation` | `workflows.phenomena` | Samples variable distributions; produces Π confidence intervals |
| `Variable` | `elements.parameter` | Dimensional parameter with formula, setpoints, and category |
| `Coefficient` | `elements.parameter` | Dimensionless Π group with symbolic expression and evaluation methods |
| `PiSolver` | `dimensional` | Internal: implements four-step Buckingham Pi algorithm |
| `validate_variables` | `validations` | Decorator: enforces structural constraints on variable sets |
| `parse_dims` | `serialization` | Parses dimensional formula string to exponent vector |
| `render_latex` | `serialization` | Renders Pi expression string to LaTeX format |

---

## Parameter Attribute Reference

All `Variable` attributes accepted as plain dict keys:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `_idx` | `int` | Yes | Unique integer index within the variable set |
| `_sym` | `str` | Yes | Symbolic name, e.g. `'\\rho'` or `'velocity'` |
| `_fwk` | `str` | Yes | Schema identifier: `'PHYSICAL'`, `'COMPUTATION'`, `'SOFTWARE'`, or `'CUSTOM'` |
| `_cat` | `str` | Yes | Category: `'IN'`, `'OUT'` (exactly one), or `'CTRL'` |
| `relevant` | `bool` | Yes | `True` to include in analysis; `False` to exclude |
| `_dims` | `str` | Yes | Dimensional formula using schema FDUs, e.g. `'M*L^-3'` |
| `_setpoint` | `float` | Recommended | Nominal value in natural units |
| `_std_setpoint` | `float` | Recommended | Standardised value for `calculate_setpoint()` evaluation |
| `distribution_params` | `dict` | Optional | Statistical distribution for Monte Carlo simulation |

---

*This report documents the public API of PyDASA v0.7.0. All docstrings use Google style and are Sphinx-compatible.*
