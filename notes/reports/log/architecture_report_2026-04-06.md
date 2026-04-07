# PyDASA Software Architecture Report

**Date:** 2026-04-06
**Version:** 0.7.0
**Status:** Development Status 3 -- Alpha
**Licence:** GPL-3.0-or-later
**Repository:** https://github.com/DASA-Design/PyDASA
**Documentation:** https://pydasa.readthedocs.io/en/latest/

---

## 1. System Overview

### 1.1 Introduction

Dimensional analysis is a foundational method in science and engineering for deriving relationships between physical quantities based on their dimensional structure. The Buckingham Pi theorem, formalized by Vaschy and Buckingham, provides the mathematical framework for reducing a set of dimensioned variables into a smaller set of dimensionless coefficients (Pi groups). Despite its significance, few software tools exist that implement the complete dimensional analysis pipeline, from variable definition through matrix construction, symbolic solving, sensitivity analysis, and Monte Carlo simulation, within a single library.

PyDASA (Python Dimensional Analysis for Science and Architecture) addresses this gap. It is a Python library that implements the Buckingham Pi theorem across multiple dimensional domains. While the traditional domain is classical physics (Length, Mass, Time, Temperature, Electric Current, Amount of Substance, Luminous Intensity), PyDASA extends dimensional analysis to computational science (Time, Space, Complexity) and software architecture (Time, Data, Structure, Entropy, Connectivity), as well as arbitrary user-defined dimensional frameworks.

The library targets researchers and engineers who perform dimensional analysis as part of their investigative or design workflow. Its development follows the theoretical foundations established by H. Gortler in *Dimensionalanalyse: Eine Theorie der physikalischen Dimensionen mit Anwendungen*. The library is currently in Alpha status (version 0.7.0), with active development of the core analysis pipeline and post-analysis capabilities.

### 1.2 Purpose and Audience

PyDASA serves as a computational tool for:

1. Defining physical, computational, or domain-specific variables with dimensional formulas.
2. Constructing and solving the dimensional matrix using the Buckingham Pi theorem.
3. Generating dimensionless coefficients (Pi groups) from the nullspace of the matrix.
4. Deriving new coefficients from algebraic combinations of computed Pi groups.
5. Performing symbolic and numerical sensitivity analysis on dimensionless coefficients.
6. Running Monte Carlo simulations to quantify uncertainty in coefficient behavior.

The intended audience includes scientific researchers, engineering practitioners, and graduate students in fields where dimensional analysis is applicable. The library presupposes familiarity with the Buckingham Pi theorem and basic linear algebra.

### 1.3 Chapter Roadmap

This report is structured following the Attribute-Driven Design (ADD) method of Bass, Clements, and Kazman (3rd edition). Section 2 derives functional requirements from the implemented code. Section 3 presents the utility tree of quality attribute scenarios. Section 4 documents design principles and their trade-offs. Section 5 provides architectural views (context, internal structure, data flow, process, configuration). Section 6 covers implementation details including dependencies, CI/CD pipeline, and the release process. Section 7 identifies limitations and future work.

---

## 2. Functional Requirements

The following functional requirements are derived from the actual implementation in the `src/pydasa/` source tree. Each requirement maps to one implemented capability.

**FR-1: Dimensional Framework Definition.** The system shall support four dimensional frameworks (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM), each defining a set of Fundamental Dimensional Units (FDUs) with symbols, units, names, and precedence ordering. The CUSTOM framework shall allow user-defined FDU sets.
*Traced to:* `core/setup.py` (Frameworks enum), `dimensional/vaschy.py` (Schema class), `core/cfg/default.json`.

**FR-2: Variable Definition.** The system shall represent variables with dimensional formulas (e.g., `L*T^-1`), LaTeX symbol notation, category classification (IN, OUT, CTRL), numerical bounds (original and standardized units), and statistical distribution specifications. Variables shall validate dimensional expressions against the active Schema's regex patterns at construction time.
*Traced to:* `elements/parameter.py` (Variable), `elements/specs/conceptual.py`, `elements/specs/symbolic.py`, `elements/specs/numerical.py`, `elements/specs/statistical.py`.

**FR-3: Dimensional Matrix Construction.** The system shall construct a dimensional matrix (FDUs x Variables) from a set of relevant variables, sorting variables by category (IN, OUT, CTRL), filtering by relevance flag, and extracting working FDUs from variable dimension strings.
*Traced to:* `dimensional/model.py` (Matrix.create_matrix).

**FR-4: Matrix Solving via Buckingham Pi Theorem.** The system shall compute the Row-Reduced Echelon Form (RREF) of the dimensional matrix using SymPy, identify pivot columns, compute the nullspace, and generate one dimensionless Coefficient (Pi group) per nullspace vector.
*Traced to:* `dimensional/model.py` (Matrix.solve_matrix, Matrix._generate_coefficients).

**FR-5: Coefficient Derivation.** The system shall derive new dimensionless coefficients from algebraic expressions over existing Pi groups, supporting multiplication, division, exponentiation, addition, subtraction, and numeric constant multipliers. Derived coefficients shall be marked with category DERIVED.
*Traced to:* `dimensional/model.py` (Matrix.derive_coefficient), `serialization/parser.py` (parse_dim_expr).

**FR-6: Coefficient Setpoint Calculation.** The system shall compute the numerical setpoint value of each dimensionless coefficient from variable setpoint values and their dimensional exponents.
*Traced to:* `dimensional/buckingham.py` (Coefficient.calculate_setpoint).

**FR-7: Symbolic Sensitivity Analysis.** The system shall perform symbolic sensitivity analysis by computing partial derivatives of dimensionless coefficient expressions with respect to each variable, evaluating them at specified operating points using SymPy differentiation and lambdify.
*Traced to:* `analysis/scenario.py` (Sensitivity.analyze_symbolically).

**FR-8: Numerical Sensitivity Analysis.** The system shall perform numerical sensitivity analysis using the Fourier Amplitude Sensitivity Test (FAST) via SALib, sampling variable ranges and computing first-order and total-order sensitivity indices.
*Traced to:* `analysis/scenario.py` (Sensitivity.analyze_numerically).

**FR-9: Monte Carlo Simulation.** The system shall execute Monte Carlo simulations for dimensionless coefficients, sampling variable values from user-defined probability distributions (uniform, normal, triangular, exponential, lognormal, custom), computing coefficient values for each sample, and aggregating statistical results (mean, median, standard deviation, min, max). A shared cache mechanism shall avoid redundant variable sampling across coefficients.
*Traced to:* `analysis/simulation.py` (MonteCarlo), `workflows/practical.py` (MonteCarloSimulation).

**FR-10: LaTeX Expression Parsing.** The system shall parse LaTeX mathematical expressions into SymPy symbolic expressions, create bidirectional mappings between LaTeX notation and Python-compatible symbol names, and support nested subscripts up to five levels of brace nesting.
*Traced to:* `serialization/parser.py` (parse_latex, create_latex_mapping, extract_latex_vars), `validations/patterns.py` (LATEX_VAR_TOKEN_RE).

**FR-11: JSON Serialization.** The system shall serialize and deserialize all core entities (Variable, Coefficient, Schema, Matrix, workflow objects) to and from JSON-compatible dictionaries via `to_dict()` / `from_dict()` methods. A generic `load()` / `save()` I/O layer shall handle file-based persistence.
*Traced to:* `core/io.py`, `to_dict()` and `from_dict()` methods on Variable, Coefficient, Schema, Matrix, WorkflowBase, AnalysisEngine, SensitivityAnalysis, MonteCarloSimulation.

**FR-12: Workflow Orchestration.** The system shall provide three high-level workflow classes that compose variable management, schema configuration, and execution logic: AnalysisEngine (dimensional analysis), SensitivityAnalysis (sensitivity analysis), and MonteCarloSimulation (Monte Carlo simulation). Each workflow shall expose `reset()` and `clear()` methods for state management.
*Traced to:* `workflows/phenomena.py`, `workflows/influence.py`, `workflows/practical.py`, `workflows/basic.py` (WorkflowBase).

**FR-13: Custom Data Structures.** The system shall provide generic data structure implementations (ArrayList, SingleLinkedList, Node types, SCHashTable with Bucket and MapEntry) for internal use. These structures shall support customizable comparison functions and memory-optimized slot allocation.
*Traced to:* `structs/lists/arlt.py`, `structs/lists/sllt.py`, `structs/lists/ndlt.py`, `structs/tables/scht.py`, `structs/tables/htme.py`, `structs/tools/`.

---

## 3. Critical Quality Attributes (Utility Tree)

The following utility tree identifies the Architecturally Significant Requirements (ASRs) that shaped PyDASA's design. Primary quality attributes are marked with an asterisk (*).

```
Utility
|
+-- INTEROPERABILITY *
|   +-- Scientific Ecosystem Integration
|   |   +-- [H/M] A Variable exported via to_dict() is used to construct a pandas
|   |   |         DataFrame in 3 lines with no custom adapter code.
|   |   +-- [H/M] A Coefficient's pi_expr rendered as LaTeX is consumed by
|   |   |         matplotlib's mathtext engine without manual post-processing.
|   |   +-- [M/L] Matrix._dim_mtx returned as a NumPy NDArray is directly usable
|   |             in scipy.linalg operations without conversion.
|   +-- Domain Generality
|       +-- [H/H] A user defines a CUSTOM framework with 3 user-defined FDUs;
|       |         the entire analysis pipeline (matrix, RREF, coefficients) runs
|       |         identically to the PHYSICAL framework without code changes.
|       +-- [M/M] A COMPUTATION framework analysis uses the same Schema, Variable,
|                 and Matrix classes as PHYSICAL, differing only in JSON config data.
|
+-- MAINTAINABILITY *
|   +-- Schema Extensibility
|   |   +-- [H/M] A developer adds a new dimensional framework by supplying a JSON
|   |   |         config entry in default.json; no changes to Schema, Variable, or
|   |   |         Matrix source code are required.
|   |   +-- [M/L] A new FDU is added to PHYSICAL by editing default.json; the regex
|   |             engine regenerates patterns automatically via Schema._setup_regex().
|   +-- Separation of Concerns
|       +-- [H/M] A developer modifies the Variable's numerical bounds logic
|       |         (NumericalSpecs) without affecting its dimensional logic
|       |         (SymbolicSpecs) or classification logic (ConceptualSpecs).
|       +-- [M/M] A new analysis method (e.g., Sobol indices) is added by creating
|                 a new class inheriting from Foundation; no changes to the
|                 dimensional pipeline modules are required.
|
+-- TESTABILITY *
|   +-- Module Independence
|   |   +-- [H/M] The Sensitivity class is independently importable and fully
|   |   |         testable by providing a LaTeX expression string, without
|   |   |         instantiating a full Matrix or AnalysisEngine.
|   |   +-- [H/M] A Dimension object is independently testable with from_dict()
|   |   |         and to_dict() round-trip validation, without loading the
|   |   |         full framework configuration.
|   |   +-- [M/L] Each validation decorator (validate_type, validate_range, etc.)
|   |             is independently testable by applying it to a trivial setter.
|   +-- Regression Oracles
|       +-- [H/M] The standard PHYSICAL framework with 7 FDUs produces a known
|       |         RREF matrix for a predefined variable set; this is used as a
|       |         regression oracle for the full pipeline.
|       +-- [M/L] A Variable's _prepare_dims() on input "L*T^-1" produces exactly
|                 [1, 0, -1, 0, 0, 0, 0] for PHYSICAL; this is an oracle for
|                 the dimensional column generation.
|
+-- PERFORMANCE
|   +-- Computation Efficiency
|   |   +-- [M/M] A 7-variable PHYSICAL analysis (7 FDUs) completes matrix
|   |   |         creation, RREF computation, and coefficient generation in under
|   |   |         2 seconds on a standard laptop with no external compute.
|   |   +-- [M/H] A Monte Carlo simulation with 10,000 experiments for 3
|   |             coefficients sharing 5 variables uses a shared cache to avoid
|   |             redundant sampling, completing in under 10 seconds.
|   +-- Memory Efficiency
|       +-- [L/M] Custom data structures use Python 3.10+ __slots__ via the
|                 alloc_slots() decorator to reduce per-instance memory overhead.
|
+-- USABILITY
|   +-- API Ergonomics
|   |   +-- [M/L] A researcher can define variables, run dimensional analysis, and
|   |   |         obtain Pi coefficients by following a 5-step workflow using
|   |   |         AnalysisEngine without reading the full API reference.
|   |   +-- [M/L] LaTeX symbol notation (e.g., \\rho_{1}) is accepted directly as
|   |             variable symbols; the parser handles conversion internally.
|   +-- Error Diagnostics
|       +-- [M/M] A Variable constructed with an invalid dimensional expression
|                 raises a ValueError at instantiation time identifying the
|                 offending expression and the valid FDU regex pattern.
|
+-- AVAILABILITY
    +-- Input Validation
        +-- [H/M] A Coefficient constructed with mismatched variable list and
        |         dimensional column lengths raises a ValueError before any matrix
        |         computation begins, preventing silent numerical errors.
        +-- [M/M] A Matrix with zero output variables raises a ValueError at
                  _update_variable_stats() time, before matrix construction,
                  with a message specifying the constraint.
```

The primary quality attributes (Interoperability, Maintainability, Testability) were chosen because PyDASA must integrate with the broader scientific Python ecosystem, support multiple dimensional domains without code duplication, and maintain correctness guarantees through its mathematical pipeline. Performance is secondary because typical analysis problems involve small matrices (fewer than 10 variables).

---

## 4. Design Principles

### 4.1 Schema Generality

**Principle.** The dimensional analysis engine is parameterized by a Schema object that defines the active set of Fundamental Dimensional Units. All downstream processing (regex validation, matrix construction, coefficient generation) derives its configuration from the Schema rather than hardcoded constants.

The trade-off is increased initialization complexity: every Schema instantiation triggers FDU setup, regex compilation, precedence validation, and symbol map construction. A simpler design could hardcode the seven SI base dimensions, but this would preclude the COMPUTATION, SOFTWARE, and CUSTOM frameworks that distinguish PyDASA from single-domain tools.

*Traced to:* `dimensional/vaschy.py` (Schema), `core/cfg/default.json`.

### 4.2 Validation-First Construction

**Principle.** All domain entities (Dimension, Variable, Coefficient, Matrix) perform comprehensive validation during `__post_init__()`, using a decorator-based validation system on property setters. Invalid state is rejected at construction time rather than during computation.

The trade-off is verbosity in the validation decorator stack (e.g., a single property setter may carry `@validate_type`, `@validate_emptiness`, `@validate_choices`, and `@validate_pattern`), and a performance cost at object construction time. The benefit is that once an object is constructed, downstream code can trust its invariants without defensive checks.

*Traced to:* `validations/decorators.py` (9 reusable decorators), `core/basic.py` (SymBasis, IdxBasis, Foundation).

### 4.3 Compositional Variable Design

**Principle.** The Variable class is composed from four independent specification classes, each representing a distinct philosophical perspective: ConceptualSpecs (identity), SymbolicSpecs (notation), NumericalSpecs (values), and StatisticalSpecs (distributions). These are combined via Python multiple inheritance in a single dataclass.

The trade-off is the complexity of the MRO (Method Resolution Order) and the need for explicit `clear()` delegation across all four parent classes. The benefit is that each perspective can evolve independently: modifying how distributions work (StatisticalSpecs) does not require touching dimensional formula logic (SymbolicSpecs).

*Traced to:* `elements/parameter.py` (Variable), `elements/specs/` package.

### 4.4 Interoperability by Default

**Principle.** All core entities expose `to_dict()` / `from_dict()` serialization, NumPy arrays for matrix operations, SymPy expressions for symbolic computation, and LaTeX strings for typesetting. The library uses standard scientific Python types (NDArray, sp.Matrix, sp.Expr) as its internal representations rather than custom wrappers.

The trade-off is a dependency on four heavyweight libraries (numpy, scipy, sympy, matplotlib), even for users who only need the dimensional matrix computation. The benefit is zero-friction integration with pandas DataFrames, matplotlib rendering, and scipy statistical functions.

*Traced to:* `dimensional/model.py` (_dim_mtx as NDArray, _sym_mtx as sp.Matrix), `dimensional/buckingham.py` (_data as NDArray), `serialization/parser.py` (parse_latex wrapping sympy.parsing.latex).

### 4.5 Separation of Pipeline Stages

**Principle.** The dimensional analysis pipeline is decomposed into explicit, independently invocable stages: variable preparation, matrix construction, RREF solving, coefficient generation, and optional derivation. Each stage has a dedicated method on the Matrix class (`_prepare_analysis()`, `create_matrix()`, `solve_matrix()`, `_generate_coefficients()`, `derive_coefficient()`). Workflow classes (AnalysisEngine, SensitivityAnalysis, MonteCarloSimulation) compose these stages.

The trade-off is that users must understand the stage ordering if they bypass the convenience methods (`run_analysis()`, `run_simulation()`). The benefit is that intermediate results (the dimensional matrix, the RREF, the pivot columns) are inspectable for educational, debugging, and reporting purposes.

*Traced to:* `dimensional/model.py` (Matrix), `workflows/phenomena.py` (AnalysisEngine.run_analysis).

### 4.6 Configuration-Driven Behavior

**Principle.** Framework definitions, FDU metadata, variable categories, coefficient categories, and analysis modes are defined in external JSON configuration (`core/cfg/default.json`) and loaded at module initialization through a frozen singleton (`PyDASAConfig`). Enums provide type-safe access to configuration values.

The trade-off is that the frozen-singleton pattern makes runtime reconfiguration impossible without restarting the Python process. The benefit is immutability and consistency: once the configuration is loaded, all modules share the same canonical definitions.

*Traced to:* `core/setup.py` (PyDASAConfig, Frameworks, VarCardinality, CoefCardinality, AnaliticMode, SimulationMode), `core/constants.py`, `core/cfg/default.json`.

---

## 5. Architecture Views

### 5.1 Diagram Index

| # | View | Description |
|---|------|-------------|
| 1 | Context Diagram | External actors and system boundary |
| 2 | Internal Structure | Module dependency graph with responsibilities |
| 3 | Data Flow | Trace from Variable definition to Coefficient output |
| 4 | Process Diagram | Dimensional analysis pipeline stages |
| 5 | Configuration View | JSON schema-driven behavior |

### 5.2 Context Diagram

```
[Diagram Placeholder: Context Diagram (C4 Level 1)]

Caption: PyDASA system context showing external actors and interactions.

Center: PyDASA Library (v0.7.0)

External Actors (Runtime):
  - Researcher / Engineer [User]
      --> Defines Variables with dimensional formulas, categories, bounds
      --> Invokes AnalysisEngine.run_analysis()
      --> Invokes SensitivityAnalysis.analyze_symbolic/numeric()
      --> Invokes MonteCarloSimulation.run_simulation()
      <-- Receives Dict[str, Coefficient] with Pi expressions and data
      <-- Receives sensitivity indices (symbolic derivatives, FAST indices)
      <-- Receives Monte Carlo statistics (mean, dev, min, max)

  - pandas [Data Processing]
      <-- Variable.to_dict() / Coefficient.to_dict() output consumed as
          DataFrame rows; Matrix.dim_mtx consumed as array data

  - matplotlib [Visualization]
      <-- Coefficient._pi_expr consumed as LaTeX mathtext strings
      <-- Simulation _data arrays plotted via standard pyplot API

  - NumPy / SciPy [Numerical Computation]
      <-- Matrix._dim_mtx (NDArray) used for array operations
      <-- scipy.stats used internally by MonteCarlo for distribution sampling
      <-- SALib (built on NumPy/SciPy) used for FAST sensitivity analysis

  - SymPy [Symbolic Computation]
      <-- Matrix._sym_mtx (sp.Matrix) used for RREF and nullspace
      <-- Sensitivity._sym_func (sp.Expr) used for symbolic differentiation
      <-- sympy.parsing.latex used to parse LaTeX expressions

External Actors (Development-time only):
  - GitHub Actions [CI/CD]
      --> Runs pytest across Python 3.10, 3.11, 3.12
      --> Runs python-semantic-release for version management
      --> Publishes to PyPI via gh-action-pypi-publish

  - PyPI [Package Registry]
      <-- Receives built wheel/sdist on release

  - Read the Docs [Documentation Hosting]
      --> Builds Sphinx documentation from docs/source/
      --> Publishes HTML, PDF, and EPUB formats

  - Codecov [Coverage Reporting]
      <-- Receives coverage.xml from pytest-cov
```

### 5.3 Internal Structure (Module Dependency Graph)

```
[Diagram Placeholder: Module/Component Diagram]

Caption: PyDASA internal package structure with dependency arrows pointing
from dependent to dependency. Each box shows package name and primary
responsibility.

Packages (top-level: src/pydasa/):

+------------------------------------------------------------------+
|  __init__.py  (Public API Surface)                                |
|  Exports: Variable, Dimension, Schema, Matrix, Coefficient,      |
|           Sensitivity, MonteCarlo, AnalysisEngine,                |
|           SensitivityAnalysis, MonteCarloSimulation,              |
|           load, save, ArrayList, SingleLinkedList,                |
|           Node, SLNode, DLNode, MapEntry, Bucket, SCHashTable    |
+------------------------------------------------------------------+
        |
        v
+------------------+    +------------------+    +--------------------+
| workflows/       |    | analysis/        |    | serialization/     |
| Orchestration    |    | Post-analysis    |    | LaTeX parsing      |
|                  |    |                  |    |                    |
| WorkflowBase     |--->| Sensitivity      |    | parse_latex()      |
| AnalysisEngine   |--->| MonteCarlo       |    | create_latex_map() |
| SensitivityAnal. |    |                  |    | parse_dim_expr()   |
| MonteCarloSim.   |    +--------+---------+    | latex_to_python()  |
+--------+---------+             |              +--------+-----------+
         |                       |                       |
         v                       v                       v
+------------------+    +------------------+    +--------------------+
| dimensional/     |    | elements/        |    | validations/       |
| Core DA engine   |    | Variable model   |    | Validation system  |
|                  |    |                  |    |                    |
| Dimension (FDU)  |    | Variable         |    | decorators.py      |
| Schema (FDU set) |    | specs/:          |    |   validate_type    |
| Coefficient (Pi) |<---| ConceptualSpecs  |    |   validate_range   |
| Matrix (solver)  |    | SymbolicSpecs    |    |   validate_pattern |
|                  |    | NumericalSpecs   |    |   validate_choices |
+--------+---------+    | StatisticalSpecs |    |   etc. (9 total)   |
         |              +--------+---------+    | error.py           |
         |                       |              | patterns.py        |
         v                       v              +--------+-----------+
+------------------+                                     ^
| core/            |                                     |
| Foundation layer |<------------------------------------+
|                  |
| basic.py:        |    +--------------------+
|   SymBasis       |    | structs/           |
|   IdxBasis       |    | Custom ADTs        |
|   Foundation     |    |                    |
| setup.py:        |    | lists/: ArrayList, |
|   PyDASAConfig   |    |   SingleLinkedList |
|   Enums          |    | tables/: SCHash,   |
| io.py: load/save |    |   Bucket, MapEntry |
| constants.py     |    | tools/: hashing,   |
+--------+---------+    |   math, memory     |
         |              | types/: generics,  |
         v              |   functions        |
+------------------+    +--------------------+
| core/cfg/        |
|   default.json   |
+------------------+

+--------------------+
| context/           |
| [STUB - not impl.] |
|                    |
| conversion.py     |
| system.py         |
| units.py          |
+--------------------+

Dependency flow summary:
  workflows/ --> dimensional/, analysis/, serialization/, core/
  analysis/  --> dimensional/, elements/, serialization/, core/
  dimensional/ --> elements/, core/, serialization/, validations/
  elements/  --> core/, serialization/, validations/
  serialization/ --> validations/
  core/      --> validations/ (via basic.py), core/cfg/
  structs/   --> (self-contained, no domain dependencies)
  context/   --> (stubs, no active dependencies)
```

### 5.4 Data Flow Diagram

```
[Diagram Placeholder: Information Flow Diagram]

Caption: Data flow trace for a typical dimensional analysis, from Variable
definition through to Coefficient output. Numbered steps show the sequence.

1. USER defines Variables as dataclass instances:
   Variable(_sym="V", _dims="L*T^-1", _cat="IN", _fwk="PHYSICAL",
            relevant=True, _std_setpoint=10.0, ...)

2. Variable.__post_init__() triggers:
   2a. ConceptualSpecs.__post_init__() --> creates default Schema for PHYSICAL
   2b. SymbolicSpecs._prepare_dims():
       - _standardize_dims("L*T^-1") --> "L^(1)*T^(-1)"  [regex-based]
       - _sort_dims("L^(1)*T^(-1)")  --> "L^(1)*T^(-1)"  [FDU precedence]
       - _setup_sympy(...)            --> "L**(1)* T**(-1)" [SymPy format]
       - _setup_column(...)           --> [1, 0, -1, 0, 0, 0, 0] [dim_col]

3. USER passes Dict[str, Variable] to AnalysisEngine or Matrix:
   engine = AnalysisEngine(_fwk="PHYSICAL")
   engine.variables = {"V": v1, "d": v2, ...}

4. AnalysisEngine.create_matrix() creates Matrix:
   Matrix._prepare_analysis():
   4a. _update_variable_stats() --> counts IN, OUT, CTRL; validates constraints
   4b. _sort_by_category() --> orders variables: IN first, then OUT, then CTRL
   4c. _find_output_variable() --> identifies the OUT variable
   4d. _extract_fdus() --> extracts working FDU symbols from dimension strings

5. Matrix.create_matrix():
   5a. For each relevant Variable, reads _dim_col
   5b. Fills _dim_mtx (NDArray, shape: n_fdus x n_vars)
   5c. Creates _dim_mtx_trans (transposed view)

6. Matrix.solve_matrix():
   6a. Converts _dim_mtx to _sym_mtx (sp.Matrix)
   6b. Computes RREF: _rref_mtx, _pivot_cols = _sym_mtx.rref()
   6c. Computes _nullspace = _sym_mtx.nullspace()

7. Matrix._generate_coefficients():
   7a. For each nullspace vector i:
       - Extracts vector as float list (dim_col for coefficient)
       - Creates Coefficient(_idx=i, _sym="\\Pi_{i}", _variables=relevance_lt,
                             _dim_col=vector, _pivot_lt=pivot_cols)
   7b. Coefficient.__post_init__():
       - _build_expression(var_syms, dim_col) --> LaTeX pi_expr (e.g., "\\frac{V*d}{\\nu}")
       - Stores var_dims: Dict[str, int] mapping variable symbols to exponents

8. OUTPUT: engine.coefficients returns Dict[str, Coefficient]
   Each Coefficient contains:
   - _pi_expr: LaTeX symbolic expression
   - var_dims: variable-to-exponent mapping
   - _dim_col: nullspace vector
   - to_dict(): JSON-serializable representation
```

### 5.5 Process Diagram (Dimensional Analysis Pipeline)

```
[Diagram Placeholder: UML Activity Diagram]

Caption: Complete dimensional analysis pipeline process showing the sequence
of stages, decision points, and error paths.

START
  |
  v
[Define Variables] -- User creates Variable instances with _dims, _cat, _fwk
  |
  v
<Variables valid?> --NO--> [Raise ValueError: invalid dims/cat/fwk]
  |YES
  v
[Initialize AnalysisEngine] -- Set framework, create Schema
  |
  v
[Set engine.variables] -- Triggers _convert_to_objects() if needed
  |
  v
[engine.create_matrix()] -- Creates Matrix with _schema and _variables
  |
  v
[Matrix._prepare_analysis()]
  |
  +---> _update_variable_stats()
  |       |
  |       v
  |     <n_out == 0?> --YES--> [Raise ValueError: no output variable]
  |       |NO
  |       v
  |     <n_out > 1?> --YES--> [Raise ValueError: max 1 output]
  |       |NO
  |       v
  |     <n_in == 0?> --YES--> [Raise ValueError: no input variables]
  |       |NO
  |
  +---> _sort_by_category() -- Orders: IN, OUT, CTRL
  +---> _find_output_variable()
  +---> _extract_fdus() -- Identifies active FDU symbols
  |
  v
[engine.solve()]
  |
  +---> Matrix.create_matrix()
  |       Build NDArray (n_fdus x n_vars) from variable _dim_col vectors
  |
  +---> Matrix.solve_matrix()
  |       |
  |       +---> Convert to SymPy Matrix
  |       +---> Compute RREF and pivot columns
  |       +---> Compute nullspace vectors
  |
  +---> Matrix._generate_coefficients()
          For each nullspace vector:
            Create Coefficient with _build_expression()
  |
  v
[Return Dict[str, Coefficient]]
  |
  v
<Derive additional coefficients?> --YES--> [engine.derive_coefficient(expr)]
  |NO                                           |
  |                                             v
  |                                      [parse_dim_expr()] -- Parse expression
  |                                             |
  |                                      [Create DERIVED Coefficient]
  |                                             |
  v                                             v
<Calculate setpoints?> --YES--> [engine.calculate_coefficients()]
  |NO                                  For each Coefficient:
  |                                    coef.calculate_setpoint(vars)
  v
END (coefficients available for sensitivity/MC workflows)
```

### 5.6 Process Diagram (Sensitivity Analysis)

```
[Diagram Placeholder: Sensitivity Analysis Activity Diagram]

Caption: Sensitivity analysis workflow for both symbolic and numerical modes.

START
  |
  v
[Initialize SensitivityAnalysis]
  |
  v
[Set variables and coefficients from prior dimensional analysis]
  |
  v
<Mode = SYM?> --YES--> [analyze_symbolic(val_type)]
  |                       |
  |                       +---> _create_analyses() -- Sensitivity per coefficient
  |                       |       For each Coefficient:
  |                       |         Parse pi_expr via parse_latex()
  |                       |         Build SymPy expression with symbol mapping
  |                       |
  |                       +---> For each analysis:
  |                       |       Get variable values (mean/min/max)
  |                       |       For each variable:
  |                       |         diff(sym_func, var)  [symbolic derivative]
  |                       |         lambdify() and evaluate at operating point
  |                       |
  |                       v
  |                    [Return Dict[str, Dict[str, float]]]
  |
  |NO
  v
[analyze_numeric(n_samples)]
  |
  +---> _create_analyses()
  +---> For each analysis:
  |       Define SALib problem (names, bounds)
  |       Generate FAST samples
  |       lambdify(sym_func) --> executable function
  |       Evaluate function at all sample points
  |       SALib.analyze.fast.analyze(problem, Y)
  |
  v
[Return Dict[str, Dict[str, Any]]] -- S1, ST indices per variable
  |
  v
END
```

### 5.7 Process Diagram (Monte Carlo Simulation)

```
[Diagram Placeholder: Monte Carlo Simulation Activity Diagram]

Caption: Monte Carlo simulation workflow showing distribution configuration,
cache management, and execution.

START
  |
  v
[Initialize MonteCarloSimulation]
  |
  v
[Set variables, coefficients, experiments count]
  |
  v
[run_simulation(iters, mode)]
  |
  +---> create_simulations()
  |       |
  |       +---> configure_distributions()
  |       |       For each Variable:
  |       |         Collect dist_type, dist_params, dist_func
  |       |         Store in _distributions dict
  |       |
  |       +---> configure_simulations()
  |               For each Coefficient:
  |                 Create MonteCarlo instance
  |                 Set coefficient, distributions, dependencies
  |                 Share _shared_cache reference
  |
  +---> _init_shared_cache()
  |       Allocate NaN arrays: (experiments, 1) per variable
  |
  +---> For each Coefficient simulation:
  |       sim.run(experiments, mode)
  |         |
  |         +---> <mode = DIST?> --> Sample from distributions
  |         |     <mode = DATA?> --> Use pre-existing Variable._data
  |         |
  |         +---> For each experiment:
  |         |       Sample/retrieve variable values
  |         |       Compute coefficient value: product(var^exp)
  |         |
  |         +---> Compute statistics (mean, median, dev, min, max)
  |
  v
[Store results per coefficient]
  |
  v
END
```

### 5.8 Configuration / Schema View

```
[Diagram Placeholder: Configuration Schema Diagram]

Caption: How JSON configuration drives runtime behavior across all framework
variants. The same engine processes PHYSICAL, COMPUTATION, SOFTWARE, and
CUSTOM frameworks through schema parameterization.

                    +------------------------+
                    |   core/cfg/default.json |
                    |                        |
                    | frameworks:            |
                    |   PHYSICAL:            |
                    |     fdus: {L,M,T,K,    |
                    |            I,N,C}      |
                    |   COMPUTATION:         |
                    |     fdus: {T,S,N}      |
                    |   SOFTWARE:            |
                    |     fdus: {T,D,S,E,A}  |
                    |   CUSTOM:              |
                    |     fdus: {}           |
                    |                        |
                    | variable_cardinality:  |
                    |   IN, OUT, CTRL        |
                    |                        |
                    | coefficient_cardinality:|
                    |   COMPUTED, DERIVED    |
                    |                        |
                    | analytic_modes:        |
                    |   SYM, NUM             |
                    +----------+-------------+
                               |
                               v
                    +------------------------+
                    |  PyDASAConfig (frozen)  |
                    |  Singleton via          |
                    |  __post_init__ load     |
                    |                        |
                    |  SPT_FDU_FWKS: dict    |
                    |  .frameworks: tuple    |
                    |  .parameter_cardinality|
                    |  .coefficient_cardinal.|
                    |  .analitic_modes       |
                    |  .simulation_modes     |
                    +----------+-------------+
                               |
              +----------------+----------------+
              |                |                |
              v                v                v
    +---------+----+ +--------+------+ +-------+-------+
    | Schema       | | Enums         | | Decorators    |
    | (per-fwk)    | |               | |               |
    |              | | Frameworks    | | @validate_    |
    | _fdu_lt:     | | VarCardinality| |   choices()   |
    |   [Dimension]| | CoefCardinality|  references   |
    | _fdu_regex   | | AnaliticMode  | |   enum values |
    | _fdu_pow_re  | | SimulationMode| |               |
    | _fdu_no_pow  | |               | |               |
    | _fdu_sym_re  | +---------------+ +---------------+
    +--------------+
         |
         | (Each framework variant uses the same Schema class
         |  with different FDU lists loaded from JSON)
         v
    +--------------------------------------------+
    | Dimensional Analysis Pipeline              |
    | (Matrix, Variable, Coefficient)            |
    | All use Schema.fdu_regex for validation    |
    | All use Schema.fdu_symbols for column gen  |
    | All use Schema.size for matrix dimensions  |
    +--------------------------------------------+
```

Framework-specific behavior summary:

| Framework | FDU Count | FDU Symbols | Base Unit Examples | Loaded From |
|-----------|-----------|-------------|-------------------|-------------|
| PHYSICAL | 7 | L, M, T, K, I, N, C | m, kg, s, K, A, mol, cd | default.json |
| COMPUTATION | 3 | T, S, N | s, bit, op | default.json |
| SOFTWARE | 5 | T, D, S, E, A | s, bit, req, err, func | default.json |
| CUSTOM | User-defined | User-defined | User-defined | Constructor args |

---

## 6. Implementation Details

### 6.1 Dependencies

#### Runtime Dependencies

| Package | Version Constraint | Role in PyDASA |
|---------|--------------------|----------------|
| `antlr4-python3-runtime` | ==4.11 | Parser runtime for LaTeX expression grammar (via SymPy) |
| `numpy` | >=1.26.4 | Dimensional matrix storage (NDArray), data arrays, Monte Carlo computation |
| `scipy` | >=1.13.0 | Statistical distributions for Monte Carlo sampling (scipy.stats) |
| `sympy` | >=1.12 | Symbolic matrix operations (RREF, nullspace), LaTeX parsing, differentiation, lambdify |
| `matplotlib` | >=3.8.0 | Visualization support for coefficient expressions and simulation results |
| `pandas` | >=2.1.0 | Data structuring for analysis results and tabular output |
| `SALib` | >=1.4.5 | Fourier Amplitude Sensitivity Test (FAST) for numerical sensitivity analysis |

#### Development Dependencies

| Package | Version Constraint | Role |
|---------|--------------------|------|
| `pytest` | >=8.1.1 | Test framework |
| `twine` | >=6.1.0 | Package upload to PyPI |
| `pytest-cov` | (installed in CI) | Coverage measurement |
| `nbval` | (installed in CI) | Jupyter notebook validation |

#### Documentation Dependencies

| Package | Role |
|---------|------|
| `sphinx` >=7.3.7 | Documentation generator |
| `pydata-sphinx-theme` >=0.14.0 | Theme for Read the Docs |
| `sphinx-autodoc-typehints` >=1.24.0 | Type hint rendering |
| `sphinx-autoapi` >=3.0.0 | Automatic API documentation |
| `myst-parser` >=2.0.0 | Markdown support in Sphinx |
| `nbsphinx` >=0.9.0 | Jupyter notebook rendering |
| `sphinx-copybutton`, `sphinx-favicon`, `sphinx-gitstamp`, `sphinx-prompt`, `sphinx-markdown-builder` | Various doc enhancements |

### 6.2 Build and Packaging

PyDASA uses setuptools as its build backend (`pyproject.toml`):

- **Build system:** `setuptools>=61.0.0` with `wheel`
- **Package layout:** `src/` layout (`package-dir = {"" = "src"}`)
- **Package data:** JSON configuration files included via `pydasa = ["core/cfg/*.json"]`
- **Dynamic version:** Read from `pydasa._version.__version__` (currently `"0.7.0"`)
- **Python requirement:** `>=3.10` (uses `match` statements, `__slots__` in dataclasses, `X | Y` type union syntax)

### 6.3 Development Environment

- **Python versions tested:** 3.10, 3.11, 3.12 (CI matrix)
- **Test runner:** pytest with `--cov=src/pydasa` and XML coverage reports
- **Coverage reporting:** Codecov integration via `codecov/codecov-action@v5`
- **Notebook testing:** `pytest --nbval-lax` on `tests/notebooks/` (continue-on-error)

### 6.4 CI/CD Pipeline

#### Test Workflow (`.github/workflows/test.yml`)

Triggered on push and pull request to `dev` and `main` branches.

1. Matrix strategy: Python 3.10, 3.11, 3.12 on `ubuntu-latest`.
2. Install: `pip install -e ".[dev]"` and `pytest-cov`.
3. Run: `pytest tests/ -v --tb=short --cov=src/pydasa --cov-report=xml --cov-report=term`.
4. Upload coverage to Codecov.
5. Run notebook validation (non-blocking).

#### Release Workflow (`.github/workflows/release.yml`)

Triggered on push to `main` (and temporarily `dev`) and version tags (`v*`).

1. Checkout with `fetch-depth: 0` for full history (required by semantic-release).
2. Install `python-semantic-release` and `build`.
3. Run `semantic-release version` (bumps version based on conventional commits) and `semantic-release publish` (creates GitHub release).
4. Build package via `python -m build`.
5. Publish to PyPI via `pypa/gh-action-pypi-publish@release/v1` with `skip-existing: true`.

### 6.5 Semantic Release Configuration

| Setting | Value |
|---------|-------|
| `branch` | `main` |
| `major_on_zero` | `false` (no major bumps while `<1.0`) |
| `allow_zero_version` | `true` |
| `commit_message` | `chore(release): {version} [skip ci]` |
| `tag_format` | `v{version}` |
| `minor_tags` | `feat` |
| `patch_tags` | `fix`, `perf` |
| `version_variables` | `src/pydasa/_version.py:__version__` |

Commit convention: conventional commits with tags `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`. A `feat` commit triggers a minor version bump; `fix` or `perf` triggers a patch bump.

### 6.6 Documentation Deployment

Read the Docs configuration (`.readthedocs.yaml`):

- **OS:** Ubuntu 22.04
- **Python:** 3.11
- **Builder:** Sphinx HTML (with PDF and EPUB output formats)
- **Configuration:** `docs/source/conf.py`
- **Fail on warning:** `false`
- **Post-install:** `linkify-it-py`

---

## 7. Limitations and Future Work

### 7.1 Functional Gaps

**Unit Conversion System (context/ package).** The `context/` package contains three stub modules: `conversion.py` (12 lines, TODO comment), `system.py` (17 lines, TODO comment), and `units.py` (15 lines, TODO comment). These are intended to provide unit conversion (e.g., km to m, kByte to bit), measurement system management, and a Measurement class. The `__init__.py` for `context/` is empty. The corresponding imports in `__init__.py` are commented out (`UnitStandarizer`, `MeasureSystem`, `Unit`). Current workaround: users must manually provide values in standardized units and set both original and standardized unit fields on Variables.

**DoubleLinkedList.** The `structs/lists/dllt.py` file exists but its import is commented out in `__init__.py`. This data structure is incomplete or not yet validated.

**Distribution Validation Completeness.** The `StatisticalSpecs.dist_params` setter validates parameters for `uniform` and `normal` distributions but does not validate parameters for `triangular`, `exponential`, `lognormal`, or `custom` distribution types. The TODO in `dist_type` setter notes this for future improvement.

**LaTeX Parser Limitations.** The `latex_to_python()` function in `serialization/parser.py` uses simple string replacement (`replace("\\", "")`) rather than proper regex conversion (the regex approach is commented out with a TODO). Complex LaTeX expressions with deeply nested subscripts may not round-trip perfectly through the alias system.

### 7.2 Validation Gaps

**Cross-Framework Validation.** While the system prevents mixing FDU symbols across frameworks (Schema validates framework consistency), there is no empirical validation that the COMPUTATION and SOFTWARE frameworks produce dimensionally meaningful results. The theoretical grounding for these non-physical frameworks is acknowledged but not yet experimentally validated.

**Numerical Stability.** The RREF computation delegates to SymPy's `rref()` method, which uses exact rational arithmetic. For very large matrices or pathological cases, the performance characteristics of the symbolic solver have not been benchmarked. The comment in `Matrix.solve_matrix()` suggests this is an area for future optimization.

**Monte Carlo Convergence.** The Monte Carlo simulation does not currently implement convergence diagnostics. Users must manually select the number of experiments. No automated stopping criteria or convergence checks are provided.

### 7.3 API Stability

PyDASA is classified as **Development Status 3 -- Alpha** in its PyPI classifiers. The `major_on_zero = false` semantic release setting means that breaking API changes will be released as minor version bumps (e.g., 0.7.0 to 0.8.0) without a major version increment. Users should pin to specific minor versions for reproducibility.

The public API surface, as defined in `__init__.py.__all__`, currently exposes 18 symbols. Several of these are internal data structures (ArrayList, SingleLinkedList, Node types, MapEntry, Bucket, SCHashTable) that may be moved to a private namespace in future releases.

The `allow_zero_version = true` setting indicates that the library is not yet committed to semantic versioning stability guarantees. The API may change between minor releases.

### 7.4 Reproducibility

To reproduce results with the current version:

```
pip install pydasa==0.7.0
```

The version string is stored in `src/pydasa/_version.py` and dynamically read by setuptools. The Git tag `v0.7.0` marks the corresponding commit.

---

## Appendix A: Public API Surface

The following symbols are exported from `pydasa.__init__` and listed in `__all__`:

| Symbol | Module | Category |
|--------|--------|----------|
| `__version__` | `_version` | Metadata |
| `Variable` | `elements.parameter` | Domain Entity |
| `Dimension` | `dimensional.fundamental` | Domain Entity |
| `Schema` | `dimensional.vaschy` | Domain Entity |
| `Matrix` | `dimensional.model` | Domain Entity |
| `Coefficient` | `dimensional.buckingham` | Domain Entity |
| `Sensitivity` | `analysis.scenario` | Analysis |
| `MonteCarlo` | `analysis.simulation` | Analysis |
| `AnalysisEngine` | `workflows.phenomena` | Workflow |
| `SensitivityAnalysis` | `workflows.influence` | Workflow |
| `MonteCarloSimulation` | `workflows.practical` | Workflow |
| `load` | `core.io` | I/O |
| `save` | `core.io` | I/O |
| `ArrayList` | `structs.lists.arlt` | Data Structure |
| `SingleLinkedList` | `structs.lists.sllt` | Data Structure |
| `Node`, `SLNode`, `DLNode` | `structs.lists.ndlt` | Data Structure |
| `MapEntry` | `structs.tables.htme` | Data Structure |
| `Bucket`, `SCHashTable` | `structs.tables.scht` | Data Structure |

## Appendix B: Class Inheritance Hierarchy

```
ABC
  +-- SymBasis (core/basic.py) -- Symbol, framework, alias management
        +-- IdxBasis -- Adds index/precedence
              +-- Foundation -- Adds name, description, __str__/__repr__
                    +-- Dimension (dimensional/fundamental.py) -- FDU entity
                    +-- Schema (dimensional/vaschy.py) -- FDU collection + regex
                    +-- Coefficient (dimensional/buckingham.py) -- Pi group
                    |     (also inherits BoundsSpecs)
                    +-- Matrix (dimensional/model.py) -- Dimensional matrix solver
                    +-- Sensitivity (analysis/scenario.py) -- Sensitivity engine
                    +-- MonteCarlo (analysis/simulation.py) -- MC engine
                    |     (also inherits BoundsSpecs)
                    +-- AnalysisEngine (workflows/phenomena.py)
                    |     (also inherits WorkflowBase)
                    +-- SensitivityAnalysis (workflows/influence.py)
                    |     (also inherits WorkflowBase)
                    +-- MonteCarloSimulation (workflows/practical.py)
                          (also inherits WorkflowBase)

Variable (elements/parameter.py):
  ConceptualSpecs (Foundation) + SymbolicSpecs + NumericalSpecs + StatisticalSpecs

NumericalSpecs:
  BoundsSpecs + StandardizedSpecs

WorkflowBase (workflows/basic.py):
  Standalone dataclass (no Foundation inheritance)
```

## Appendix C: File Count Summary

| Package | .py Files | Responsibility |
|---------|-----------|----------------|
| `core/` | 4 + 1 JSON | Foundation classes, config, I/O, constants |
| `dimensional/` | 4 | Dimension, Schema, Coefficient, Matrix |
| `elements/` | 2 + 4 specs | Variable with 4-perspective composition |
| `analysis/` | 2 | Sensitivity, MonteCarlo |
| `workflows/` | 4 | WorkflowBase, AnalysisEngine, SensitivityAnalysis, MonteCarloSimulation |
| `serialization/` | 1 | LaTeX parser and expression evaluator |
| `validations/` | 3 | Decorators, error handling, regex patterns |
| `structs/` | 10 | Custom ADTs (lists, tables, tools, types) |
| `context/` | 3 (stubs) | Future unit conversion system |
| Root | 2 | `__init__.py`, `_version.py` |
| **Total** | **~53 .py + 1 .json** | |
