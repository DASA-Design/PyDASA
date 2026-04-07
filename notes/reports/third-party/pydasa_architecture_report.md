# PyDASA: Software Architecture Report
**Version:** 0.7.0 | **Licence:** GPL-3.0 | **Date:** April 2026
**Repository:** https://github.com/DASA-Design/PyDASA
**Documentation:** https://pydasa.readthedocs.io/en/stable/

---

## 1. Introduction

> **[Cross-reference: See the DASA Research Programme chapter for the full theoretical background on dimensional analysis, the Buckingham Pi theorem, and the domain-agnostic hypothesis that motivates this work.]**

The DASA research programme establishes that dimensional analysis is a domain-independent mathematical framework applicable to physical, computational, and software architecture systems alike. Existing tools such as BuckinghamPy presuppose that fundamental dimensional units (FDUs) are the classical mechanical quantities, mass (M), length (L), and time (T), and provide no mechanism for operating in alternative domains. Users working outside the physical domain must construct bespoke scripts to manage variable definitions, dimensional matrix operations, and coefficient derivations, an approach that is error-prone and non-reproducible.

We developed **PyDASA** (*Python Dimensional Analysis for Software Architecture*) to operationalise the research programme's central claim. PyDASA provides a unified, schema-agnostic Python interface for defining variables in any dimensional schema, constructing and solving the dimensional matrix, deriving named dimensionless coefficients, and propagating uncertainty through the analysis via sensitivity analysis and Monte Carlo simulation. The library is available under the GPL-3.0 licence at https://github.com/DASA-Design/PyDASA, documented at https://pydasa.readthedocs.io, and installable via `pip install pydasa`. The current stable release is **v0.7.0**.

This report follows the Attribute-Driven Design (ADD) structure of Bass, Clements, and Kazman (3rd ed., 2013). Section 2 establishes functional requirements. Section 3 characterises quality attributes through a utility tree. Section 4 documents the design principles. Section 5 presents the architecture through a set of complementary views. Section 6 covers implementation details. Section 7 provides limitations and future development.

---

## 2. Key Functional Requirements

The primary functional requirement is grounded in the following user story from the project documentation:

> *As a researcher, engineer, or software architect analysing complex systems, I want a comprehensive dimensional analysis library implementing the Buckingham Pi theorem, so that I can systematically discover dimensionless relationships, validate models, and understand system behaviour across physical, computational, and software architecture domains.*

From this story, we derive six atomic, testable functional requirements:

**FR-1: Dimensional Schema Management.** The system shall allow users to select from three built-in schemas (`PHYSICAL`: M, L, T; `COMPUTATION`: T, S, N; `SOFTWARE`: T, D, E, C, A) and define fully custom schemas with arbitrary FDU symbol sets. Unit conversion is supported within a schema only; cross-schema conversion is intentionally out of scope, as a dimension valid in one schema may have no defined equivalent in another.

**FR-2: Variable Definition and Management.** The system shall allow users to define dimensional parameters specifying symbolic name, LaTeX representation, dimensional formula expressed in the schema's FDUs, setpoint values, variable category (input, output, or control), and optional statistical distribution parameters.

**FR-3: Dimensionless Coefficient Discovery.** The system shall apply the Buckingham Pi theorem algorithmically, constructing the dimensional matrix from the variable set, reducing it via linear row operations, and producing the *n - k* primary Π groups. It shall additionally support the derivation of named coefficients as algebraic combinations of primary groups and their numerical evaluation from stored setpoints.

**FR-4: Sensitivity Analysis.** The system shall support sensitivity analysis that perturbs individual variable setpoints and measures the resulting propagation to each Π coefficient value, enabling identification of the dominant variables in a dimensional model.

**FR-5: Monte Carlo Simulation.** The system shall support Monte Carlo simulation that samples variable distributions and computes the resulting distribution of each Π coefficient, producing confidence intervals for dimensionless relationships. The simulation shall accept both statistical distribution parameters defined on variables and pre-existing empirical datasets supplied by the researcher.

**FR-6: Results Export and Integration.** The system shall export all domain objects (`Variable`, `Coefficient`) to plain Python dictionaries via `to_dict()`, directly compatible with `pandas`, `matplotlib`, and `seaborn` without requiring custom adapters or intermediate transformation.

---

## 3. Critical Quality Attributes

We apply the utility tree method of Bass, Clements, and Kazman (3rd ed., 2013), which identifies seven quality attributes. Leaf nodes are ASR scenarios rated `[Business Importance / Technical Risk]` where H = High, M = Medium, L = Low. Primary attributes are marked (★).

```
Utility
│
├── AVAILABILITY
│   ├── [M/L] run_analysis() called with a singular dimensional matrix raises
│   │         a descriptive exception within 1 second naming the offending
│   │         variable; no silent NaN or incorrect Π group is produced
│   ├── [M/L] A Monte Carlo simulation encounters a numerical overflow on one
│   │         sample; the failure is logged, the sample is skipped, and the
│   │         remaining N-1 samples complete without interruption
│   └── [L/L] Ill-formed variable dictionaries (missing _cat, unparseable
│             _dims) are rejected at engine instantiation, before any state
│             propagates to the Pi solver
│
├── INTEROPERABILITY  ★
│   ├── [H/H] A Coefficient exported via to_dict() is used to construct a
│   │         pandas DataFrame in one line with no custom adapter or mapping
│   ├── [H/M] A variable supplied as a plain Python dict is accepted by
│   │         AnalysisEngine and produces output identical to a typed
│   │         Variable object input
│   ├── [H/H] A user switches AnalysisEngine from PHYSICAL to a CUSTOM
│   │         schema by supplying a new FDU list; valid Π groups are
│   │         produced without any other code change
│   └── [M/M] LaTeX Pi expressions serialised by the serialization module
│             render correctly in a Sphinx/ReadTheDocs build without
│             manual post-processing
│
├── MAINTAINABILITY  ★
│   ├── [H/M] A developer adds a new dimensional schema by supplying a FDU
│   │         symbol list; no changes to dimensional/, elements/, or
│   │         workflows/ are required
│   ├── [H/H] A researcher subclasses AnalysisEngine to add post-processing;
│   │         the Variable and Coefficient contracts (to_dict, pi_expr,
│   │         calculate_setpoint) remain stable and unchanged
│   ├── [M/M] A new workflow class is added under workflows/ consuming
│   │         existing Variable and Coefficient objects with no modification
│   │         to dimensional/ or elements/
│   └── [M/L] Code reduction refactoring does not break any operable
│             module's public API; all modules above 90% coverage pass
│             unchanged after refactoring
│
├── PERFORMANCE  ★
│   ├── [H/H] A 10-variable PHYSICAL analysis via run_analysis() completes
│   │         in under 1 second on a standard laptop with no GPU or
│   │         external compute
│   ├── [M/M] A 6-variable analysis including derive_coefficient() and
│   │         calculate_setpoint() completes end-to-end in under 2 seconds
│   ├── [M/H] A MonteCarloSimulation with 10,000 samples over 6 variables
│   │         completes in under 30 seconds on a standard laptop
│   └── [L/L] Switching from one schema to another does not degrade runtime
│             for existing analyses; no global state is mutated on switch
│
├── SECURITY
│   ├── [M/M] A malformed _dims string (e.g., "M**L" or empty string) raises
│   │         a named ValidationError with a descriptive message; no
│   │         unhandled exception or dimensional matrix corruption occurs
│   └── [L/L] Every PyPI release is traceable to a specific tagged commit
│             via the automated GitHub Actions release pipeline
│
├── TESTABILITY  ★
│   ├── [H/H] Each operable module (core, dimensional, elements, validations,
│   │         serialization, workflows) is independently importable and
│   │         testable without instantiating a full AnalysisEngine
│   ├── [H/M] The Reynolds number derivation produces exactly 1 Π group
│   │         equal to μ/(L·ρ·v) and Re = 1.00e+05; this analytically
│   │         known result serves as the full-pipeline regression oracle
│   ├── [M/M] A deliberately ill-defined variable set (two OUT variables,
│   │         or zero relevant variables) raises a named ValidationError,
│   │         not a generic Python exception or a silent wrong result
│   └── [M/H] GitHub Actions CI blocks PyPI release if any operable
│             module drops below 90% individual test coverage
│
└── USABILITY
    ├── [H/M] A researcher unfamiliar with dimensional analysis reproduces
    │         the Reynolds number result by following the 5-step quickstart
    │         in the README without reading the full API reference
    ├── [M/M] Variable definition requires only plain Python dicts; no
    │         custom DSL or framework import is needed before first use
    └── [H/M] When run_analysis() fails, the error message identifies the
              specific variable symbol and attribute at fault, not only
              "matrix is singular" or a Python traceback
```

The four primary attributes, interoperability, maintainability, performance, and testability, reflect the demands of a domain-agnostic research tool. Interoperability and maintainability follow from FR-1 and FR-6: the library must operate across schemas without modification and integrate with the scientific Python ecosystem. Testability is a prerequisite for research correctness; a silent wrong result in a research tool is more dangerous than an explicit failure. Performance ensures the library remains practical for iterative exploratory analysis during research.

---

## 4. Design Principles and Concerns

We designed PyDASA around five principles. Each is traceable to a primary quality attribute and involves an explicit trade-off.

**Schema Generality.** The library must not presuppose any particular set of FDUs. The choice of dimensional schema is a runtime configuration parameter to the `AnalysisEngine`, not a structural assumption embedded in the solver. This is the primary architectural expression of the research claim that dimensional analysis is domain-agnostic. Critically, unit conversion is defined within a schema only; cross-schema conversion is out of scope because a dimension valid in one schema, such as coupling degree (C) in the SOFTWARE schema, has no defined equivalent in the PHYSICAL schema. The trade-off is that comparative analyses across schemas require manual setpoint normalisation by the researcher.

**Separation of Concerns.** Variable specification (`elements/`), dimensional algebra (`dimensional/`), analysis orchestration (`workflows/`), constraint enforcement (`validations/`), and data export (`serialization/`) are separated into independent modules with strictly unidirectional dependencies. Consequently, each layer is independently testable, and extending the system with a new workflow class requires no modification to any existing module. The trade-off is that the layered boundary introduces a small degree of indirection for simple use cases.

**Correctness Through Early Validation.** A decorator-based validation system enforces structural constraints at variable definition and engine instantiation time, rather than at analysis runtime. For a research tool, a plausible but numerically incorrect result is far more damaging than an explicit failure. This principle shapes the entire error model. The trade-off is a small runtime overhead on every decorated method call, negligible at the problem sizes encountered in this research but worth noting for high-frequency simulation loops.

**Interoperability by Default.** All core domain objects expose a `to_dict()` method producing plain Python dictionaries. Variables are accepted as either native Python dictionaries or typed `Variable` objects, and the engine normalises both forms transparently. This lowers the barrier to exploratory use while supporting full programmatic composition. The trade-off is that internal attribute names (such as `pi_expr` and `_std_setpoint`) become part of the public API surface; any renaming constitutes a breaking change.

**Reproducibility Through Release Automation.** The release pipeline is fully automated via GitHub Actions and the Conventional Commits specification. A commit merged to `main` triggers semantic version bumping, `CHANGELOG.md` generation, GitHub release creation, and PyPI publication without manual steps. As a result, every installed version is traceable to a specific tagged commit. One deliberate deferral under this principle is the `context/` module (intra-schema unit conversion), shipped as a documented stub to keep the pipeline green while all operable modules maintain over 90% individual test coverage.

---

## 5. Software Architecture Views

Table 1 indexes the ten architectural diagrams for this system.

| Figure | View Type | Description |
|--------|-----------|-------------|
| 1 | Context | System boundary, external actors, and interaction types |
| 2 | Information Flow | Data movement through internal layers during analysis |
| 3 | Process: General DA Pipeline | End-to-end pipeline (contains Figures 4-7 as sub-processes) |
| 4 | Process: Stage 1 | Create or select dimensional schema |
| 5 | Process: Stage 2 | Build relevance list and dimensional matrix |
| 6 | Process: Stage 3 | Solve dimensional matrix and derive coefficients |
| 7 | Process: Stage 4 | Validate coefficients with sensitivity analysis and Monte Carlo |
| 8 | Module Map | Package layers and unidirectional dependency structure |
| 9 | Class and Domain Object Model | Variable and Coefficient attributes, methods, and relationships |
| 10 | Schema Configuration | Built-in and custom dimensional schema definitions |

*Table 1: Architecture diagram index.*

### 5.1 Context Diagram

> **[Figure 1 — Context Diagram]**
> *C4 Level-1 context diagram. PyDASA at the centre. External actors: Researcher/Engineer (primary user, runtime interaction via Python API). External systems: PyPI (package distribution, development-time only), GitHub (source, release management, CI/CD, development-time only), ReadTheDocs (documentation build, development-time only). Output consumers: pandas, matplotlib, seaborn (runtime, via to_dict()). A note box states explicitly: PyDASA has no runtime network dependencies. All interactions with PyPI, GitHub, and ReadTheDocs occur through automated pipelines at development and release time, not during analysis execution.*

PyDASA has no runtime network dependencies. All external system interactions, publication to PyPI, documentation builds on ReadTheDocs, and release creation on GitHub, occur through the automated CI/CD pipeline at development time. During analysis execution, the library operates entirely in-process using `numpy` and `sympy`.

### 5.2 Information Flow Diagram

> **[Figure 2 — Information Flow Diagram]**
> *Data flow diagram with seven labelled stages. (1) User input: variable dictionary (dict or Variable objects) enters AnalysisEngine. (2) Validations layer: structural constraints checked; ValidationError raised on failure with targeted message. (3) Elements layer: input normalised to typed Variable objects; relevance flag filtered to produce the active variable set. (4) Dimensional layer: FDU rows and variable columns arranged into dimensional matrix; core and residual sub-matrices partitioned; core reduced to identity; n-k Π groups produced as Coefficient objects. (5) Workflows layer: Coefficient objects stored in engine.coefficients; derive_coefficient() and calculate_setpoint() available for post-processing. (6) Serialization layer: pi_expr strings rendered to LaTeX on demand. (7) Output: Coefficient objects or plain dicts via to_dict() exit the system. Arrows labelled with data type at each boundary: dict/Variable, Variable[], float matrix, Coefficient[], dict.*

### 5.3 Process Diagrams — Dimensional Analysis Pipeline

> **[Figure 3 — General DA Pipeline (BPMN 2.0)]**
> *High-level process diagram. Start event: Researcher initiates analysis. Four sequential collapsed sub-process boxes: (1) Create or Select Schema, (2) Build Relevance List and Dimensional Matrix, (3) Solve Matrix and Derive Coefficients, (4) Validate Coefficients. Each sub-process references one of Figures 4-7. Decision gateway after each stage: validation failure returns researcher to the relevant earlier stage. End event: validated Coefficient objects available for export and downstream analysis.*

> **[Figure 4 — Stage 1: Create or Select Dimensional Schema (BPMN 2.0)]**
> *Swimlane: Researcher. Tasks: (1) Select built-in schema (PHYSICAL, COMPUTATION, or SOFTWARE) or choose CUSTOM. If CUSTOM: supply Python list of FDU symbol strings. Gateway: are all FDU symbols unique and non-empty? No: exception raised, return to researcher. Yes: schema instantiated. Output artifact: instantiated dimensional schema, passed to Stage 2.*

> **[Figure 5 — Stage 2: Build Relevance List and Dimensional Matrix (BPMN 2.0)]**
> *Swimlane: Researcher and AnalysisEngine. Tasks: (1) Researcher defines variables as dicts or Variable objects with _dims, _cat, relevant, _setpoint. (2) AnalysisEngine filters variables by relevant=True. Gateway: at least one relevant variable AND exactly one _cat=OUT? No: ValidationError identifying failing attribute. Yes: (3) Dimensional matrix constructed (FDUs as rows, relevant variables as columns). Output artifact: dimensional matrix, passed to Stage 3.*

> **[Figure 6 — Stage 3: Solve Dimensional Matrix and Derive Coefficients (BPMN 2.0)]**
> *Swimlane: AnalysisEngine (PiSolver). Tasks: (1) Core sub-matrix identified (repeating variables). (2) Linear row operations reduce core to identity matrix; same operations applied to residual. Gateway: core matrix singular? Yes: ValidationError naming the dimensionally dependent variable. No: (3) Residual and unity matrices combined to produce n-k primary Π Coefficient objects. (4) Researcher optionally calls derive_coefficient(expr) to create named coefficients. (5) calculate_setpoint() evaluates numerical value from _std_setpoint values. Output artifact: Coefficient objects, passed to Stage 4.*

> **[Figure 7 — Stage 4: Validate Coefficients (BPMN 2.0)]**
> *Parallel gateway splits into two paths. Path A (SensitivityAnalysis): individual variable setpoints perturbed one at a time; Π coefficient changes measured; dominant variables ranked. Path B (MonteCarloSimulation): variable distributions sampled (from distribution parameters OR pre-existing empirical dataset); Π coefficient distributions computed; confidence intervals produced. Both paths join at a synchronisation gateway. Decision gateway: does coefficient behaviour satisfy research criteria? No: return to Stage 2 to revise variable set. Yes: End event, results available for export.*

### 5.4 Module Map

> **[Figure 8 — Module Map]**
> *Layered box diagram. Bottom layer (Foundation): core/, validations/, serialization/. Middle layer (Domain Core): elements/, dimensional/. Top layer (Application): workflows/. Dashed box (Pending): context/, structs/. Arrows point upward from Foundation to Domain Core, from Domain Core to Application. No arrow points downward. Coverage annotations: all Foundation, Domain Core, and Application modules labelled ">90% individual coverage". Pending modules labelled "stub" or "partial". A footer note states: global project coverage ~80%, reflecting inclusion of pending stubs in the overall count.*

The module tree with coverage status:

```
src/pydasa/
├── core/           # Foundation: base classes, configuration, I/O     [>90%]
├── validations/    # Foundation: decorator-based constraint system     [>90%]
├── serialization/  # Foundation: LaTeX rendering, formula parsing      [>90%]
├── elements/       # Domain Core: Variable and Coefficient objects     [>90%]
├── dimensional/    # Domain Core: Buckingham Pi solver                 [>90%]
├── workflows/      # Application: AnalysisEngine, SensitivityAnalysis,
│                   #              MonteCarloSimulation                 [>90%]
├── context/        # Pending: intra-schema unit conversion             [stub]
└── structs/        # Pending: dimensional data structures              [partial]
```

Dependencies flow strictly upward from Foundation through Domain Core to Application. Nothing in the lower layers imports from the upper layers, preventing circular dependencies and ensuring that the mathematical core can be tested entirely independently of orchestration.

### 5.5 Class and Domain Object Model

> **[Figure 9 — Class and Domain Object Model (UML)]**
> *UML class diagram. Class Variable: attributes _idx: int, _sym: str, _fwk: str, _cat: str (IN|OUT|CTRL), relevant: bool, _dims: str, _setpoint: float, _std_setpoint: float, distribution_params: dict (optional); methods to_dict() -> dict. Class Coefficient: attributes pi_expr: str, symbol: str, name: str, variables: list[Variable]; methods calculate_setpoint() -> float, to_dict() -> dict. Class AnalysisEngine: attributes _fwk: str, _fdu_list: list (optional), variables: dict, coefficients: dict; methods run_analysis() -> dict, derive_coefficient(expr, symbol, name) -> Coefficient. Class SensitivityAnalysis: consumes Coefficient[]; produces sensitivity report dict. Class MonteCarloSimulation: consumes Coefficient[], accepts dataset: list (optional); produces distribution dict with confidence intervals. Relationships: AnalysisEngine aggregates Variable[] (1..*); AnalysisEngine produces Coefficient[] (0..*); SensitivityAnalysis and MonteCarloSimulation each accept Coefficient[] from AnalysisEngine without conversion.*

### 5.6 Schema Configuration Diagram

> **[Figure 10 — Schema Configuration Diagram]**
> *Four-column table diagram. Column 1 (PHYSICAL): FDUs M (Mass), L (Length), T (Time); example variable density M·L⁻³, velocity L·T⁻¹, viscosity M·L⁻¹·T⁻¹. Column 2 (COMPUTATION): FDUs T (Time), S (Storage), N (Nodes); example variables throughput T⁻¹·S, data volume S, node count N. Column 3 (SOFTWARE): FDUs T (Time), D (Dependency), E (Event), C (Coupling), A (Abstraction); example variables coupling degree C, abstraction level A, event rate T⁻¹·E. Column 4 (CUSTOM): FDUs user-defined Python list of strings; example shown as BIOLOGICAL schema with M (Mass), T (Time), C (Concentration). A single AnalysisEngine row spans all four columns with label "same solver, no code changes". Footer note: unit conversion is supported within a schema only; cross-schema comparison requires manual setpoint normalisation by the researcher.*

---

## 6. Implementation Details

### 6.1 Runtime Dependencies

| Dependency | Version Constraint | Role |
|------------|--------------------|------|
| `numpy` | >=1.21 | Dimensional matrix construction and linear algebra (row reduction) |
| `sympy` | >=1.9 | Symbolic algebra for dimensional formula parsing and Pi expression generation |

No framework-level, cloud, or domain-specific runtime dependencies are introduced. The library installs into any standard Python 3.8+ environment without conflict.

### 6.2 Development Dependencies

Development dependencies are specified under the `[dev]` extras group in `pyproject.toml` and are not installed by `pip install pydasa`:

| Dependency | Role |
|------------|------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage measurement and reporting |
| `sphinx` | Documentation build |
| `sphinx-rtd-theme` | ReadTheDocs theme |
| `conventional-changelog` | Automated changelog generation |

### 6.3 Development Process

PyDASA is developed using Scrum as the iterative framework. The release history, comprising 27 tagged releases visible on GitHub, reflects sprint-based increments. Release management follows the Conventional Commits specification:

| Commit type | Trigger | Example |
|-------------|---------|---------|
| `feat` | MINOR version bump (0.6.x → 0.7.0) | `feat(workflows): add Monte Carlo empirical dataset support` |
| `fix` | PATCH version bump (0.7.0 → 0.7.1) | `fix(dimensional): handle rank-deficient core matrix edge case` |
| `BREAKING CHANGE` footer | MAJOR version bump (0.x → 1.0.0) | redesign of Variable API |

On merge to `main`, GitHub Actions automatically analyses commit messages, computes the semantic version increment, updates `_version.py` and `pyproject.toml`, regenerates `CHANGELOG.md`, creates a GitHub release with auto-generated notes, and publishes to PyPI. No manual steps are required after merge.

---

## 7. Limitations and Future Development

### 7.1 Intra-Schema Unit Conversion

The `context/` module is currently a stub. The system correctly handles dimensional algebra within a schema but does not yet automate the conversion of numerical setpoint values between measurement systems within the same schema, for example converting between SI and imperial values within the PHYSICAL schema. Analyses requiring such conversions are handled through manual setpoint normalisation. Completing the `context/` module is the highest development priority for the next major release.

### 7.2 Data Structures

The `structs/` module has partial test coverage. Certain edge cases in complex variable dependency graphs, specifically circular statistical dependencies between variables in Monte Carlo simulation, are not yet fully validated. Researchers with highly interdependent variable sets should validate results against known analytical solutions where possible.

### 7.3 Software Schema Empirical Validation

The SOFTWARE schema (FDUs: T, D, E, C, A) is implemented and tested at the module level. Empirical validation of the Π groups it produces against real-world software architecture measurements is ongoing and constitutes a central empirical contribution of the DASA research programme. Correctness is currently validated through structural consistency: the solver produces the expected number of dimensionless groups with the expected dimensional homogeneity.

### 7.4 API Stability

PyDASA is a research tool at version 0.7.0. The API is stable within a major version, but breaking changes between major versions are expected as the research programme evolves. All results in this dissertation are reproducible using `pydasa==0.7.0`.

### 7.5 Future Development Priorities

Planned work, in order of priority:

- Complete the `context/` module with intra-schema unit conversion and configurable measurement system mappings.
- Expand `structs/` with full test coverage for complex variable dependency graphs.
- Provide a higher-level declarative API to reduce boilerplate for common analysis patterns.
- Extend the tutorial library with worked examples from the COMPUTATION and SOFTWARE schemas.

---

*All computational results referenced in this report were produced using PyDASA v0.7.0. To reproduce: `pip install pydasa==0.7.0`.*
