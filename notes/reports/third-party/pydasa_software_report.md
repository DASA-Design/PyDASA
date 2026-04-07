# PyDASA: Software Report
**Version:** 0.7.0 | **Licence:** GPL-3.0 | **Language:** Python 3.8+
**Repository:** https://github.com/DASA-Design/PyDASA
**Documentation:** https://pydasa.readthedocs.io/en/stable/
**PyPI:** `pip install pydasa`

---

## 1. Overview

PyDASA (*Python Dimensional Analysis for Software Architecture*) is an open-source Python library that implements the Buckingham Pi theorem for dimensional analysis across physical, computational, and software architecture domains. Its central contribution is a schema-agnostic solver: the same analysis engine operates across all dimensional domains without modification, enabling researchers to apply the dimensional analysis methodology to problems that existing tools such as BuckinghamPy and custom bespoke scripts cannot address.

The library is the primary computational tool of the DASA (*Dimensional Analysis for Software Architecture*) research programme. It provides a unified interface for the full dimensional analysis workflow: defining dimensional variables in any schema, constructing and solving the dimensional matrix, deriving named dimensionless coefficients, and propagating statistical uncertainty through the model via sensitivity analysis and Monte Carlo simulation. All results are exportable to standard scientific Python libraries (`pandas`, `matplotlib`, `seaborn`) without custom adapters.

PyDASA is available under the GPL-3.0 licence, installable from PyPI, documented on ReadTheDocs, and maintained under a fully automated CI/CD release pipeline on GitHub. The current stable release is **v0.7.0**.

---

## 2. Problem Statement

The Buckingham Pi theorem establishes that any system of *n* variables expressed in *k* independent fundamental dimensional units (FDUs) can be fully characterised by *n - k* dimensionless groups (Π coefficients), independent of scale or unit system. This theorem is well-established in classical physics and engineering, where it underpins similarity analysis, experimental design, and model scaling.

Existing software tools, however, implement dimensional analysis exclusively within the physical domain. BuckinghamPy assumes that FDUs are mass (M), length (L), and time (T). SymPy provides the linear algebra primitives but no dimensional analysis abstraction. No existing tool supports alternative dimensional schemas for computational or software architecture domains, or provides integrated uncertainty quantification within a single dimensional analysis pipeline.

Researchers working outside the physical domain must write bespoke scripts to manage variable definitions, dimensional matrix construction, and coefficient derivation. These scripts are typically specific to a single analysis, difficult to reproduce, and not transferable across problems or domains. PyDASA addresses this gap by providing a reusable, schema-agnostic, fully tested library for the complete dimensional analysis workflow.

---

## 3. Core Capabilities

### 3.1 Dimensional Schema Management

PyDASA provides three built-in dimensional schemas and a mechanism for user-defined custom schemas:

| Schema | FDUs | Intended Domain |
|--------|------|-----------------|
| `PHYSICAL` | M (Mass), L (Length), T (Time) | Classical mechanics, fluid dynamics, heat transfer |
| `COMPUTATION` | T (Time), S (Storage), N (Nodes) | Computational systems, distributed architectures |
| `SOFTWARE` | T (Time), D (Dependency), E (Event), C (Coupling), A (Abstraction) | Software architecture analysis |
| `CUSTOM` | User-defined list | Any domain with well-defined dimensional quantities |

The schema is a runtime parameter to the `AnalysisEngine`. Switching schemas requires no code change beyond updating the `_fwk` argument. Unit conversion is supported within a schema only; cross-schema comparison requires manual setpoint normalisation, as dimensions valid in one schema may have no defined equivalent in another.

### 3.2 Variable Definition

Variables are defined as Python dictionaries or typed `Variable` objects. Each variable carries:

- Symbolic name and LaTeX representation
- Dimensional formula expressed in the schema's FDU symbols (e.g., `"M*L^-3"` for density)
- Category: `IN` (input), `OUT` (output, exactly one required), or `CTRL` (control)
- Relevance flag: `relevant=True` includes the variable in the analysis; `False` excludes it without removing it from the definition
- Nominal setpoint values (`_setpoint`, `_std_setpoint`) for numerical evaluation
- Optional statistical distribution parameters for Monte Carlo simulation

Both input forms, plain dicts and `Variable` objects, are accepted transparently by the engine.

### 3.3 Dimensionless Coefficient Discovery

The `AnalysisEngine.run_analysis()` method executes the Buckingham Pi algorithm in four steps:

1. Filter variables by `relevant=True`
2. Construct the dimensional matrix (FDUs as rows, variables as columns) and partition it into core and residual sub-matrices
3. Reduce the core sub-matrix to an identity matrix via linear row operations
4. Combine the reduced residual and unity matrices to produce the *n - k* primary Π coefficients

The `derive_coefficient()` method then allows researchers to construct named dimensionless numbers as algebraic combinations of primary groups (for example, recovering the Reynolds number as the reciprocal of the primary group). The `calculate_setpoint()` method evaluates any coefficient numerically from the stored variable setpoints.

### 3.4 Sensitivity Analysis

The `SensitivityAnalysis` workflow accepts `Coefficient` objects from `AnalysisEngine` and systematically perturbs each variable's standardised setpoint. For each perturbation, the resulting change in each Π coefficient value is measured. This identifies the dominant variables in a dimensional model and quantifies the analysis's robustness to measurement uncertainty.

### 3.5 Monte Carlo Simulation

The `MonteCarloSimulation` workflow propagates statistical uncertainty through the dimensional model by sampling variable distributions over a configurable number of trials. It accepts two input modes:

- **Distribution parameters** defined on variables (mean, standard deviation, distribution type)
- **Pre-existing empirical datasets** supplied as lists of observed values, enabling validation against experimental measurements

The simulation produces the mean, standard deviation, and configurable confidence intervals for each Π coefficient, along with optional access to the raw sample array.

### 3.6 Export and Integration

All domain objects (`Variable`, `Coefficient`) expose a `to_dict()` method returning plain Python dictionaries. These dictionaries are directly compatible with `pandas` DataFrame construction, `matplotlib` and `seaborn` plotting, and JSON serialisation, without any custom adapter or schema mapping. LaTeX rendering of Pi expressions is provided by the `serialization` module for direct use in Sphinx documentation builds.

---

## 4. Installation

### Requirements

- Python 3.8 or later
- `numpy` >= 1.21
- `sympy` >= 1.9

No framework-level, cloud, or domain-specific runtime dependencies.

### Standard Installation

```bash
pip install pydasa
```

### Version-Pinned Installation (for dissertation reproducibility)

```bash
pip install pydasa==0.7.0
```

### Verify Installation

```python
import pydasa
print(pydasa.__version__)   # 0.7.0
```

### Development Installation

```bash
git clone https://github.com/DASA-Design/PyDASA.git
cd PyDASA
pip install -e ".[dev]"
pytest tests/
```

---

## 5. Quick Start

### Example A: Reynolds Number (PHYSICAL Schema)

The Reynolds number Re = (ρ · v · L) / μ is the standard validation example in the PyDASA documentation because the expected result is analytically known. For the parameter values below, Re = 1.00 × 10⁵, placing the flow in the turbulent regime.

```python
from pydasa.workflows.phenomena import AnalysisEngine

# Step 1 — Define variables
variables = {
    "\\rho": {
        "_idx": 0, "_sym": "\\rho", "_fwk": "PHYSICAL",
        "_cat": "IN", "relevant": True,
        "_dims": "M*L^-3",          # density: kg/m³
        "_setpoint": 1000.0, "_std_setpoint": 1000.0,
    },
    "v": {
        "_idx": 1, "_sym": "v", "_fwk": "PHYSICAL",
        "_cat": "OUT",              # exactly ONE output required
        "relevant": True,
        "_dims": "L*T^-1",          # velocity: m/s
        "_setpoint": 2.0, "_std_setpoint": 2.0,
    },
    "L": {
        "_idx": 2, "_sym": "L", "_fwk": "PHYSICAL",
        "_cat": "IN", "relevant": True,
        "_dims": "L",               # length: m
        "_setpoint": 0.05, "_std_setpoint": 0.05,
    },
    "\\mu": {
        "_idx": 3, "_sym": "\\mu", "_fwk": "PHYSICAL",
        "_cat": "IN", "relevant": True,
        "_dims": "M*L^-1*T^-1",     # dynamic viscosity: Pa·s
        "_setpoint": 0.001, "_std_setpoint": 0.001,
    },
}

# Step 2 — Create and run the engine
engine = AnalysisEngine(_idx=0, _fwk="PHYSICAL")
engine.variables = variables
results = engine.run_analysis()
# Output: Number of dimensionless groups: 1
#         Pi_0: mu/(L*rho*v)

# Step 3 — Derive the Reynolds number as the reciprocal of Pi_0
pi_0_key = list(engine.coefficients.keys())[0]
Re = engine.derive_coefficient(
    expr=f"1/{pi_0_key}", symbol="Re", name="Reynolds Number"
)
print(f"Re = {Re.calculate_setpoint():.2e}")
# Output: Re = 1.00e+05  →  Turbulent flow regime

# Step 4 — Export for downstream analysis
import pandas as pd
df = pd.DataFrame([list(engine.coefficients.values())[0].to_dict()])
```

### Example B: Custom Schema (BIOLOGICAL Domain)

This example demonstrates the custom schema capability with a simplified pharmacokinetics problem. Three FDUs are defined: mass (M), time (T), and concentration (C).

```python
from pydasa.workflows.phenomena import AnalysisEngine

# Step 1 — Define a custom BIOLOGICAL schema
custom_fdu_list = ["M", "T", "C"]   # mass, time, concentration

# Step 2 — Define variables in the custom schema
variables = {
    "dose": {
        "_idx": 0, "_sym": "dose", "_fwk": "CUSTOM",
        "_cat": "IN", "relevant": True,
        "_dims": "M",               # administered mass: kg
        "_setpoint": 0.01, "_std_setpoint": 0.01,
    },
    "clearance": {
        "_idx": 1, "_sym": "CL", "_fwk": "CUSTOM",
        "_cat": "IN", "relevant": True,
        "_dims": "M*T^-1",          # mass cleared per unit time: kg/s
        "_setpoint": 0.002, "_std_setpoint": 0.002,
    },
    "concentration": {
        "_idx": 2, "_sym": "C", "_fwk": "CUSTOM",
        "_cat": "OUT", "relevant": True,
        "_dims": "C",               # measured concentration
        "_setpoint": 5.0, "_std_setpoint": 5.0,
    },
    "half_life": {
        "_idx": 3, "_sym": "t_half", "_fwk": "CUSTOM",
        "_cat": "IN", "relevant": True,
        "_dims": "T",               # biological half-life: s
        "_setpoint": 3600.0, "_std_setpoint": 3600.0,
    },
}

# Step 3 — Instantiate engine with the custom schema
engine = AnalysisEngine(_idx=0, _fwk="CUSTOM", _fdu_list=custom_fdu_list)
engine.variables = variables
results = engine.run_analysis()
# The engine processes the custom schema identically to built-in schemas.
# The same Pi solver, validation, and export mechanisms apply.
```

---

## 6. Package Structure

```
pydasa/
├── core/           # Foundation: base classes, configuration, I/O     [>90% coverage]
├── validations/    # Foundation: decorator-based constraint system     [>90% coverage]
├── serialization/  # Foundation: LaTeX rendering, formula parsing      [>90% coverage]
├── elements/       # Domain Core: Variable and Coefficient objects     [>90% coverage]
├── dimensional/    # Domain Core: Buckingham Pi solver                 [>90% coverage]
├── workflows/      # Application: AnalysisEngine, SensitivityAnalysis,
│                   #              MonteCarloSimulation                 [>90% coverage]
├── context/        # Pending: intra-schema unit conversion             [stub]
└── structs/        # Pending: dimensional data structures              [partial]
```

Global project coverage is approximately 80%, reflecting the inclusion of the two pending stub modules in the overall count. All six operable modules maintain over 90% individual test coverage.

Dependencies flow strictly upward: `workflows/` depends on `elements/` and `dimensional/`; `elements/` and `dimensional/` depend on `core/`, `validations/`, and `serialization/`. Nothing in the lower layers depends on `workflows/`.

---

## 7. Development

### Contributing

PyDASA uses Conventional Commits for automated versioning:

```bash
# Feature addition (triggers MINOR bump, e.g. 0.7.0 → 0.8.0)
git commit -m "feat(workflows): add bootstrap simulation workflow"

# Bug fix (triggers PATCH bump, e.g. 0.7.0 → 0.7.1)
git commit -m "fix(dimensional): handle rank-deficient core matrix edge case"

# Breaking change (triggers MAJOR bump, e.g. 0.7.0 → 1.0.0)
git commit -m "feat(api)!: redesign Variable attribute naming

BREAKING CHANGE: _fwk renamed to _schema across all domain objects"
```

### Release Pipeline

On merge to `main`, GitHub Actions automatically:

1. Analyses commit messages to determine the semantic version increment
2. Updates `_version.py` and `pyproject.toml`
3. Regenerates `CHANGELOG.md`
4. Creates a GitHub release with auto-generated release notes
5. Publishes the package to PyPI

No manual steps are required after merge. Every release is traceable to a specific tagged commit.

---

## 8. Limitations

**Intra-schema unit conversion.** The `context/` module is a stub. PyDASA correctly handles dimensional algebra within a schema but does not yet automate conversion of setpoint values between measurement systems within the same schema (for example, from SI to imperial units within PHYSICAL). Cross-schema comparison is intentionally out of scope. Analyses requiring unit conversion are handled through manual normalisation.

**Data structures.** The `structs/` module has partial test coverage. Edge cases involving circular statistical dependencies between variables in Monte Carlo simulation are not yet fully validated.

**SOFTWARE schema empirical validation.** The SOFTWARE schema (FDUs: T, D, E, C, A) is implemented and unit-tested. Empirical validation against real-world software architecture measurements is ongoing as part of the DASA research programme.

**API stability.** The API is stable within a major version. Breaking changes between major versions are expected as the research programme evolves. Pin to `pydasa==0.7.0` to reproduce results reported in the DASA dissertation.

---

## 9. Licence and Citation

PyDASA is released under the **GNU General Public Licence v3.0 (GPL-3.0)**.

```
DASA-Design/PyDASA. Python Dimensional Analysis for Software Architecture.
Version 0.7.0. GPL-3.0. https://github.com/DASA-Design/PyDASA, 2026.
```

For BibTeX:

```bibtex
@software{pydasa2026,
  author  = {{DASA-Design}},
  title   = {{PyDASA}: Python Dimensional Analysis for Software Architecture},
  version = {0.7.0},
  year    = {2026},
  licence = {GPL-3.0},
  url     = {https://github.com/DASA-Design/PyDASA},
}
```

---

## 10. Resources

| Resource | URL |
|----------|-----|
| Source code | https://github.com/DASA-Design/PyDASA |
| Documentation | https://pydasa.readthedocs.io/en/stable/ |
| PyPI | https://pypi.org/project/pydasa/ |
| Issue tracker | https://github.com/DASA-Design/PyDASA/issues |
| Changelog | https://github.com/DASA-Design/PyDASA/blob/main/CHANGELOG.md |

---

*All results in the DASA research programme were produced using PyDASA v0.7.0. To reproduce: `pip install pydasa==0.7.0`.*
