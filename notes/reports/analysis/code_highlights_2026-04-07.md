# PyDASA Code Highlights Report

**Date:** 2026-04-07  
**Version:** 0.7.0  
**Scope:** `src/pydasa/` -- Architectural patterns, key design decisions, and clever engineering

---

## 1. FDU Singleton Management

PyDASA treats the Fundamental Dimensional Units as configuration data, not as hard-coded constants. The entire set of frameworks -- Physical, Computation, Software, and Custom -- lives in a single JSON file that acts as the ground truth for the library. A frozen singleton loads this file exactly once and exposes it as an immutable object for the entire runtime. This means there is no possibility of accidental mutation, no duplicate loading, and one unambiguous place to change dimensional definitions.

### The JSON Source of Truth

The file `core/cfg/default.json` defines every framework, its FDUs, their symbols, units, and descriptions:

```json
// core/cfg/default.json (lines 1-43)
{
    "frameworks": {
    "PHYSICAL": {
        "name": "Physical Framework",
        "description": "Traditional physical dimensional framework (e.g., Length, Mass, Time).",
        "fdus": {
        "L": { "unit": "m",   "name": "Length",              "description": "Distance between two points in space." },
        "M": { "unit": "kg",  "name": "Mass",                "description": "Amount of matter in an object." },
        "T": { "unit": "s",   "name": "Time",                "description": "Duration of an event or interval." },
        "K": { "unit": "K",   "name": "Temperature",         "description": "Measure of average kinetic energy of particles." },
        "I": { "unit": "A",   "name": "Electric Current",    "description": "Flow of electric charge." },
        "N": { "unit": "mol", "name": "Amount of Substance", "description": "Quantity of entities (e.g., atoms, molecules)." },
        "C": { "unit": "cd",  "name": "Luminous Intensity",  "description": "Perceived power of light in a given direction." }
        }
    },
    "COMPUTATION": {
        "name": "Computation Framework",
        "fdus": {
        "T": { "unit": "s",   "name": "Time",       "description": "..." },
        "S": { "unit": "bit", "name": "Space",      "description": "..." },
        "N": { "unit": "op",  "name": "Complexity", "description": "..." }
        }
    },
    ...
    }
}
```

This is not a passive data file. The JSON drives regex generation, matrix row ordering, validation rules, and symbolic processing -- everything downstream reads from this single source.

### The Frozen Singleton

`core/setup.py` loads the JSON through a frozen dataclass with a classic singleton pattern:

```python
# core/setup.py (lines 176-210)
@dataclass(frozen=True)
class PyDASAConfig:
    """Singleton class for PyDASA configuration."""

    _instance: ClassVar["PyDASAConfig | None"] = None

    SPT_FDU_FWKS: dict = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization to load default configuration from file."""
        module_dir = Path(__file__).parent
        fp = module_dir / DFLT_CFG_FOLDER / DFLT_CFG_FILE
        cfg_data = load(fp)

        # Since the dataclass is frozen, use object.__setattr__ to set attributes
        object.__setattr__(self,
                           "SPT_FDU_FWKS",
                           cfg_data.get("frameworks", {}))

    @classmethod
    def get_instance(cls) -> "PyDASAConfig":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

The `frozen=True` flag makes the dataclass immutable after creation -- any attempt to reassign an attribute raises `FrozenInstanceError`. The `object.__setattr__` call inside `__post_init__` is the one legal bypass: it sets the data during construction before the freeze takes effect. After that, the configuration is locked.

The module-level instantiation ensures every import gets the same object:

```python
# core/setup.py (line 260)
PYDASA_CFG: PyDASAConfig = PyDASAConfig()
```

**What this enables downstream:** Every Schema, Variable, and Matrix in the library reads framework definitions from `PYDASA_CFG.SPT_FDU_FWKS`. When someone adds a new FDU to the JSON, every component picks it up automatically -- no code changes needed.

---

## 2. Validation Decorators

PyDASA uses a decorator-based validation system that separates *what to check* from *what the property does*. Instead of scattering `if not isinstance(...)` checks across hundreds of setter methods, each validation concern is captured as a composable decorator. Setters become clean one-liners, and the validation logic is written once, tested once, and reused everywhere.

### The Decorator Definitions

The core decorator is `validate_type`, which handles type checking, None tolerance, and NaN tolerance in a single wrapper:

```python
# validations/decorators.py (lines 28-95)
def validate_type(*expected_types: type,
                  allow_none: bool = True,
                  allow_nan: bool = False) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, value: Any) -> Any:
            if value is None:
                if not allow_none:
                    raise ValueError(f"{func.__name__} cannot be None.")
                return func(self, value)

            if isinstance(value, float) and np.isnan(value):
                if not allow_nan:
                    raise ValueError(f"{func.__name__} cannot be np.nan.")
                return func(self, value)

            if not isinstance(value, expected_types):
                type_names = " or ".join(t.__name__ for t in expected_types)
                raise ValueError(f"{func.__name__} must be {type_names}, got {type(value).__name__}.")
            return func(self, value)
        return wrapper
    return decorator
```

The `validate_range` decorator supports both static bounds and dynamic bounds that read from the object at call time -- enabling cross-attribute validation like "mean must be between min and max":

```python
# validations/decorators.py (lines 299-386)
def validate_range(min_value: Optional[float] = None,
                   max_value: Optional[float] = None,
                   min_attr: Optional[str] = None,
                   max_attr: Optional[str] = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, value: Any) -> Any:
            if value is not None:
                # Check static bounds
                if min_value is not None and value < min_value:
                    raise ValueError(...)
                # Check dynamic bounds from sibling attributes
                if min_attr and hasattr(self, min_attr):
                    min_val = getattr(self, min_attr)
                    if min_val is not None and value < min_val:
                        raise ValueError(...)
            return func(self, value)
        return wrapper
    return decorator
```

### How They Compose on Real Setters

The decorators stack cleanly. Here is a real example from `elements/specs/numerical.py` where `mean` must be a number, must allow NaN, and must fall within the min-max range:

```python
# elements/specs/numerical.py (lines 155-169)
@mean.setter
@validate_type(int, float, allow_none=False, allow_nan=True)
@validate_range(min_attr="_min", max_attr="_max")
def mean(self, val: Optional[float]) -> None:
    self._mean = val
```

And from `core/basic.py`, where the symbol setter validates type, rejects empty strings, and checks the LaTeX regex pattern -- all in three decorator lines:

```python
# core/basic.py (lines 75-88)
@sym.setter
@validate_type(str)
@validate_emptiness()
@validate_pattern(LATEX_RE, allow_alnum=True)
def sym(self, val: str) -> None:
    self._sym = val
```

The `validate_choices` decorator is used with Enum types to constrain properties to valid framework or category values:

```python
# core/basic.py (lines 99-111)
@fwk.setter
@validate_type(str)
@validate_choices(PYDASA_CFG.frameworks)
def fwk(self, val: str) -> None:
    self._fwk = val
```

**What this enables downstream:** Adding a new validated property to any class requires zero boilerplate -- just stack the right decorators. The validation rules are declarative: you read the setter signature and immediately know the constraints. Errors are consistent, tracing the property name and the violation, across the entire library.

---

## 3. LaTeX Parser / Serialization

PyDASA works with LaTeX symbols natively -- variables are written as `\rho_{req}`, `\mu_{1}`, or `M_{a*(c*t_{R_{P*(A*(C*S))}})}` and the library must parse these accurately for symbolic math, Python aliasing, and dimensional expression evaluation. This is harder than it sounds: LaTeX subscripts can nest arbitrarily, and naive regex fails on anything beyond one level. The parser solves this with a layered brace-matching regex and a two-pass extraction pipeline.

### The Nested Brace Regex

The key challenge is matching subscripts like `M_{buf_{AS}}` or deeply nested expressions. The `patterns.py` module builds a regex that supports up to 5 levels of brace nesting by stacking non-recursive patterns:

```python
# validations/patterns.py (lines 27-46)
_BRACE_L0: str = r"[^{}]*"
_BRACE_L1: str = r"(?:[^{}]|\{" + _BRACE_L0 + r"\})*"
_BRACE_L2: str = r"(?:[^{}]|\{" + _BRACE_L1 + r"\})*"
_BRACE_L3: str = r"(?:[^{}]|\{" + _BRACE_L2 + r"\})*"
_BRACE_L4: str = r"(?:[^{}]|\{" + _BRACE_L3 + r"\})*"

LATEX_VAR_TOKEN_RE: str = (
    r"(\\[A-Za-z]+|[A-Za-z][A-Za-z0-9]*)"
    r"(?:_(?:[A-Za-z0-9]+|\{" + _BRACE_L4 + r"\}))?"
)
```

Each `_BRACE_L` level wraps the previous one: L0 matches non-brace characters, L1 matches content that may contain one level of braces, L2 two levels, and so on. The final `LATEX_VAR_TOKEN_RE` captures a LaTeX command (like `\rho`) or a plain identifier, followed by an optional subscript that can contain up to 5 levels of nested braces. This is a pragmatic alternative to a full parser -- it covers all realistic scientific notation without the weight of a grammar.

### The Extraction Pipeline

`serialization/parser.py` uses this regex to extract variables from LaTeX expressions and build bidirectional mappings between LaTeX and Python symbols:

```python
# serialization/parser.py (lines 50-82)
def extract_latex_vars(expr: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Pair the variable names in LaTeX format with their Python equivalents."""
    # Extract latex variable names with regex (supports nested subscripts)
    matches = [m.group(0) for m in re.finditer(LATEX_VAR_TOKEN_RE, expr)]

    # Filter out ignored LaTeX commands
    matches = [m for m in matches if m not in IGNORE_EXPR]

    # Create mappings both ways
    latex_to_py = {}
    py_to_latex = {}

    for m in matches:
        latex_var = m
        py_var = m.lstrip("\\")
        py_var = py_var.replace("_{", "_")
        py_var = py_var.replace("}", "")

        latex_to_py[latex_var] = py_var
        py_to_latex[py_var] = latex_var

    return latex_to_py, py_to_latex
```

The `create_latex_mapping` function goes further, creating SymPy symbols from the regex-extracted variables rather than relying on `parse_latex`'s `free_symbols`. This is a deliberate design choice documented in the code:

```python
# serialization/parser.py (lines 105-107)
    # We don't rely on parse_latex's free_symbols because it treats multi-letter
    # subscripts like \rho_{req} as multiplication: rho_r * e * q
```

The function also includes a fallback alias mechanism for cases where SymPy's rendering of a symbol does not match the Python alias, preventing lookup failures during `.subs()` operations.

**What this enables downstream:** Variables can be defined with full LaTeX notation, displayed beautifully in notebooks and reports, yet still be used in Python computation through automatic aliasing. The Pi coefficient expression builder and sensitivity analysis engines all consume the Python aliases, while the user-facing outputs render LaTeX.

---

## 4. Matrix Construction and RREF Solving

The core of dimensional analysis is the dimensional matrix and its solution via the Buckingham Pi theorem. PyDASA constructs this matrix from Variable objects, solves it using Row-Reduced Echelon Form (RREF), and extracts the nullspace to produce dimensionless Pi groups. The matrix is the bridge between the physics (what dimensions each variable has) and the algebra (which combinations are dimensionless).

### Matrix Construction

The `create_matrix` method in `dimensional/model.py` builds the matrix by stacking each variable's dimensional column. Each column is a vector of FDU exponents -- for example, velocity (L*T^-1) becomes `[1, 0, -1, 0, 0, 0, 0]` in the physical framework:

```python
# dimensional/model.py (lines 428-485)
def create_matrix(self) -> None:
    """Builds the dimensional matrix."""
    if not self._relevance_lt:
        raise ValueError("No relevant variables to create matrix from.")

    # Prepare dimensional columns for variables that need it
    for var in self._relevance_lt.values():
        if (not var._dim_col or len(var._dim_col) == 0) and var._dims:
            if not var._schema or var._schema.fwk != self._fwk:
                var._schema = self._schema
            var._prepare_dims()

    # Get dimensions
    n_fdu = len(self._schema.fdu_symbols)
    n_var = len(self._relevance_lt)

    # Initialize empty matrix
    self._dim_mtx = np.zeros((n_fdu, n_var), dtype=float)

    # Fill matrix with dimension columns
    for var in self._relevance_lt.values():
        dim_col = var._dim_col
        if len(dim_col) < n_fdu:
            dim_col = dim_col + [0] * (n_fdu - len(dim_col))
        elif len(dim_col) > n_fdu:
            dim_col = dim_col[:n_fdu]
        self._dim_mtx[:, var._idx] = dim_col

    # Create transposed version
    self._dim_mtx_trans = self._dim_mtx.T
```

The method handles mismatched column lengths (padding with zeros or truncating), assigns schemas to variables that lack one, and stores both the original and transposed forms.

### RREF Solving and Nullspace Extraction

The `solve_matrix` method delegates the heavy algebra to SymPy's `rref()` and `nullspace()`:

```python
# dimensional/model.py (lines 487-510)
def solve_matrix(self) -> None:
    """Solves the dimensional matrix via RREF and nullspace."""
    if not isinstance(self._dim_mtx, np.ndarray) or self._dim_mtx.size == 0:
        self.create_matrix()

    # Convert to SymPy for symbolic computation
    self._sym_mtx = sp.Matrix(self._dim_mtx)

    # Compute RREF and pivot columns
    rref_result, pivot_cols = self._sym_mtx.rref()

    # Store results
    self._rref_mtx = np.array(rref_result).astype(float)
    self._pivot_cols = list(pivot_cols)

    # Generate coefficients from nullspace
    self._generate_coefficients()
```

**The math in brief:** RREF (Gaussian elimination to reduced row echelon form) identifies which variables are "dependent" (pivot columns) and which are "free." The nullspace of the dimensional matrix contains all vectors of exponents that produce dimensionless combinations. Each nullspace vector becomes one Pi group.

### Coefficient Generation from the Nullspace

Each nullspace vector is converted into a `Coefficient` object with its own LaTeX expression:

```python
# dimensional/model.py (lines 512-557)
def _generate_coefficients(self) -> None:
    """Creates Coefficient objects from each nullspace vector."""
    self._nullspace = self._sym_mtx.nullspace()
    self._coefficients.clear()
    var_syms = [var for var in self._relevance_lt.keys()]

    for i, vector in enumerate(self._nullspace):
        vector_np = np.array(vector).flatten().astype(float)
        pi_sym = f"\\Pi_{{{i}}}"
        coef = Coefficient(
            _idx=i,
            _sym=pi_sym,
            _alias=f"Pi_{i}",
            _fwk=self._fwk,
            _cat=CoefCardinality.COMPUTED.value,
            _variables=self._relevance_lt,
            _dim_col=vector_np.tolist(),
            _pivot_lt=self._pivot_cols,
            _name=f"Pi-{i}",
            description=f"Dimensionless coefficient {i} from nullspace"
        )
        self._coefficients[pi_sym] = coef
```

The `Coefficient.__post_init__` then calls `_build_expression` to produce a LaTeX fraction from the exponent vector -- positive exponents go to the numerator, negative to the denominator:

```python
# dimensional/buckingham.py (lines 197-247)
def _build_expression(self, var_lt, dim_col) -> tuple[str, dict]:
    numerator = []
    denominator = []
    parameters = {}

    for sym, exp in zip(var_lt, dim_col):
        if exp > 0:
            part = sym if exp == 1 else f"{sym}^{{{exp}}}"
            numerator.append(part)
        elif exp < 0:
            part = sym if exp == -1 else f"{sym}^{{{-exp}}}"
            denominator.append(part)
        else:
            continue
        parameters[sym] = exp

    num_str = "1" if not numerator else "*".join(numerator)
    if not denominator:
        return num_str, parameters
    else:
        denom_str = "*".join(denominator)
        return f"\\frac{{{num_str}}}{{{denom_str}}}", parameters
```

**What this enables downstream:** The `analyze()` method chains all three steps -- prepare, create, solve -- into a single call. The resulting Pi groups are fully formed objects with LaTeX expressions, numerical evaluation methods (`calculate_setpoint`), and data generation for simulation. Derived coefficients can be composed from computed ones using the expression parser from Section 3.

---

## 5. Schema-Driven Architecture

The Schema object is the bridge between the static JSON configuration and the live analysis engine. It takes a framework name (PHYSICAL, COMPUTATION, SOFTWARE, or CUSTOM), loads the corresponding FDUs from the singleton config, and generates framework-specific regex patterns for validation. The same analysis code runs identically across all frameworks -- only the Schema changes.

### Schema Initialization from Config

When a Schema is created with `_fwk="PHYSICAL"`, `__post_init__` reads the Physical FDU definitions from the singleton and constructs `Dimension` objects:

```python
# dimensional/vaschy.py (lines 112-125)
def __post_init__(self) -> None:
    super().__post_init__()
    self._setup_fdus()
    self._validate_fdu_precedence()
    self._update_fdu_map()
    self._update_fdu_symbols()
    self._setup_regex()
```

```python
# dimensional/vaschy.py (lines 216-242)
def _setup_default_framework(self) -> List[Dimension]:
    """Returns the default FDU precedence list for the specified framework."""
    _dflt_fwk_map = PYDASA_CFG.get_instance().SPT_FDU_FWKS
    _fwk_map: Dict[str, Any] = cast(Dict[str, Any], _dflt_fwk_map)
    ans = []
    if self.fwk in _fwk_map:
        fdus_dict = _fwk_map[self.fwk].get("fdus", {})
        for idx, (sym, data) in enumerate(fdus_dict.items()):
            fdu = Dimension(
                _idx=idx,
                _sym=sym,
                _fwk=self._fwk,
                _unit=data.get("unit", ""),
                _name=data.get("name", ""),
                description=data.get("description", ""))
            ans.append(fdu)
    return ans
```

### Dynamic Regex Generation

After loading FDUs, the Schema generates regex patterns tailored to the active framework. For the Physical framework with symbols `[L, M, T, K, I, N, C]`, it produces a regex that matches only those symbols in dimensional expressions:

```python
# dimensional/vaschy.py (lines 273-304)
def _setup_regex(self) -> None:
    if not self._fdu_lt:
        return None

    _fdu_str = ''.join(self.fdu_symbols)

    # Main regex: e.g., ^[LMTKINC](\^-?\d+)?(\*[LMTKINC](?:\^-?\d+)?)*$
    self._fdu_regex = rf"^[{_fdu_str}](\^-?\d+)?(\*[{_fdu_str}](?:\^-?\d+)?)*$"

    self._fdu_pow_regex = DFLT_POW_RE

    # No-power regex: matches FDU symbols NOT followed by ^
    self._fdu_no_pow_regex = rf"[{_fdu_str}](?!\^)"

    # Symbol regex for extraction
    self._fdu_sym_regex = rf"[{_fdu_str}]"
```

A Computation framework Schema produces completely different regexes -- `^[TSN](\^-?\d+)?...` -- from the same code path. This is how framework-agnostic analysis works: the Schema parameterizes all dimensional validation.

### Custom Framework Support

The `_setup_fdus` method routes to custom handling when `_fwk == "CUSTOM"`:

```python
# dimensional/vaschy.py (lines 129-163)
def _setup_fdus(self) -> None:
    if self.fwk in PYDASA_CFG.get_instance().frameworks and self.fwk != Frameworks.CUSTOM.value:
        if not self._fdu_lt:
            self._fdu_lt = self._setup_default_framework()
        else:
            self._convert_fdu_lt()
    elif self.fwk == Frameworks.CUSTOM.value:
        if not self._fdu_lt:
            raise ValueError("Custom framework requires '_fdu_lt' to define FDUs")
        self._convert_fdu_lt()
```

Custom frameworks skip the JSON lookup entirely and use user-provided Dimension objects, which are validated and converted through `_convert_fdu_lt`. This means a researcher can define their own dimensional system (economic units, biological units, etc.) and use the full analysis pipeline with zero library modifications.

**What this enables downstream:** The Matrix class stores a single `_schema: Schema` attribute. Every dimensional validation, regex match, column ordering, and FDU lookup goes through that Schema. Switching from Physical to Computation analysis is a one-line change at construction time.

---

## 6. Compositional Variable Design

A Variable in PyDASA is not a monolithic class with dozens of unrelated attributes. It is composed from four independent perspectives, each answering a different question about the variable. This separation means that each concern is developed, tested, and documented independently, yet they combine through multiple inheritance into a single object.

### The Four Perspectives

```python
# elements/parameter.py (lines 37-95)
@dataclass
class Variable(ConceptualSpecs, SymbolicSpecs, NumericalSpecs, StatisticalSpecs):
    """A comprehensive implementation that combines Parameter and Variable functionality.

    This class composes four philosophical perspectives through multiple inheritance:
    - ConceptualSpecs: Identity and classification (what IS this variable?)
    - SymbolicSpecs: Mathematical representation (how do we WRITE it?)
    - NumericalSpecs: Computational values (what VALUES can it take?)
    - StatisticalSpecs: Probabilistic modeling (how do we MODEL uncertainty?)
    """
```

### ConceptualSpecs -- "What IS this variable?"

Manages category (INPUT/OUTPUT/CTRL), framework reference, and relevance:

```python
# elements/specs/conceptual.py (lines 34-71)
@dataclass
class ConceptualSpecs(Foundation):
    """Conceptual perspective: variable identity and classification."""

    _schema: Optional[Schema] = None
    _cat: str = VarCardinality.IN.value
    relevant: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        # If no schema provided, create default based on framework
        if self._schema is None and self._fwk != Frameworks.CUSTOM.value:
            self._schema = Schema(_fwk=self._fwk)
```

The auto-provisioning of a Schema when none is provided means a Variable can be created with just `Variable(_fwk="PHYSICAL", _dims="L*T^-1")` and it will have a fully functional dimensional framework behind it.

### SymbolicSpecs -- "How do we WRITE it?"

Handles dimensional expressions, standardization, sympy conversion, and column extraction:

```python
# elements/specs/symbolic.py (lines 36-86)
@dataclass
class SymbolicSpecs:
    """Symbolic perspective: mathematical representation."""

    _dims: str = ""              # e.g., "L*T^-1"
    _units: str = ""             # e.g., "m/s"
    _std_dims: Optional[str] = None   # e.g., "L^(1)*T^(-1)"
    _sym_exp: Optional[str] = None    # e.g., "L**(1)* T**(-1)"
    _dim_col: List[int] = field(default_factory=list)  # e.g., [1, 0, -1, ...]
    _std_units: str = ""
```

The `_prepare_dims` method chains four transformations: standardize (add explicit exponents) -> sort (by FDU precedence) -> setup sympy (replace `^` with `**`) -> setup column (extract integer exponents into a list):

```python
# elements/specs/symbolic.py (lines 101-109)
def _prepare_dims(self) -> None:
    self._std_dims = self._standardize_dims(self._dims)
    self._std_dims = self._sort_dims(self._std_dims)
    self._sym_exp = self._setup_sympy(self._std_dims)
    self._dim_col = self._setup_column(self._sym_exp)
```

### NumericalSpecs -- "What VALUES can it take?"

Composes `BoundsSpecs` (original units) with `StandardizedSpecs` (standardized units) and adds discretization:

```python
# elements/specs/numerical.py (lines 425-469)
@dataclass
class NumericalSpecs(BoundsSpecs, StandardizedSpecs):
    """Numerical perspective: computational value ranges."""

    _step: Optional[float] = 1e-3
    _data: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
```

The `BoundsSpecs` and `StandardizedSpecs` are themselves separate dataclasses, so a variable tracks original and standardized values independently. The `min` setter uses `@validate_range(max_attr="_max")` to enforce `min <= max` dynamically.

### StatisticalSpecs -- "How do we MODEL uncertainty?"

Carries distribution type, parameters, a callable sampling function, and dependency tracking:

```python
# elements/specs/statistical.py (lines 25-54)
@dataclass
class StatisticalSpecs:
    """Statistical perspective: probabilistic distributions."""

    _dist_type: str = "uniform"
    _dist_params: Optional[Dict[str, Any]] = field(default_factory=dict)
    _depends: List[str] = field(default_factory=list)
    _dist_func: Optional[Callable[..., float]] = None
```

The `sample()` method delegates to the user-provided `_dist_func`, and `_depends` tracks inter-variable relationships for calculated variables like `F = m * a`.

### Integration in Variable

The Variable class itself is thin -- it inherits everything and only adds `__post_init__` orchestration, `clear()` delegation, and serialization:

```python
# elements/parameter.py (lines 97-129)
def __post_init__(self) -> None:
    super().__post_init__()
    if not self._sym:
        self._sym = f"V_{self._idx}" if self._idx >= 0 else "V_{}"
    if self._schema and len(self._dims) > 0 and self._dims != "n.a.":
        _schema = self._schema
        if not self._validate_exp(self._dims, _schema.fdu_regex):
            raise ValueError(...)
        self._prepare_dims()
    if not self._alias:
        self._alias = latex_to_python(self._sym)
```

**What this enables downstream:** Each perspective can evolve independently. A new distribution type in `StatisticalSpecs` does not touch dimensional processing. Serialization (`to_dict` / `from_dict`) traverses all four perspectives uniformly through dataclass field introspection.

---

## 7. Foundation Layer — The Class Hierarchy Root

Every domain object in PyDASA — Variable, Coefficient, Dimension, Schema — inherits from the same three-level base class chain in `core/basic.py`. This is the skeleton that gives every entity in the library a consistent shape: a symbol, a framework, an index, a name, and validated setters for all of them. Without this, each domain class would reinvent its own symbol handling, its own framework validation, and its own string representation.

### SymBasis — Symbol and Framework Identity

The root is an abstract class that gives every entity a LaTeX symbol (`_sym`), a framework context (`_fwk`), and a Python alias (`_alias`). The setters use the validation decorators from section 2, so every entity in the library validates its symbol against the LaTeX regex and its framework against the singleton's known frameworks:

```python
# core/basic.py (lines 34-65)
@dataclass
class SymBasis(ABC):
    """Abstract Class to manage the entity's symbolic, sorting,
    and dimensional domain/framework functionalities."""

    _sym: str = ""
    _fwk: str = Frameworks.PHYSICAL.value
    _alias: str = ""
```

The `fwk` setter demonstrates how the singleton and decorators compose at the foundation level — `PYDASA_CFG.frameworks` is the set of valid framework names loaded from JSON, and `validate_choices` rejects anything not in that set:

```python
# core/basic.py (lines 99-111)
@fwk.setter
@validate_type(str)
@validate_choices(PYDASA_CFG.frameworks)
def fwk(self, val: str) -> None:
    self._fwk = val
```

### IdxBasis — Ordering and Precedence

The second layer adds an integer index (`_idx`) that determines ordering in the dimensional matrix. This is how FDUs maintain their precedence (L before M before T) and how Variables keep their column position. The index is validated to be a non-negative integer through the `validate_index` decorator:

```python
# core/basic.py (lines 145-185)
@dataclass
class IdxBasis(SymBasis):
    """Basic class to manage index/precedence functionalities."""

    _idx: int = -1

    @idx.setter
    @validate_type(int, allow_none=False)
    @validate_index()
    def idx(self, val: int) -> None:
        self._idx = val
```

### Foundation — The Complete Base

The third layer adds `_name` and `description` (human-readable metadata) plus a `__str__` that introspects all dataclass fields for debugging. This is the class that `Dimension`, `Coefficient`, and `Variable` (via `ConceptualSpecs`) all inherit from:

```python
# core/basic.py (lines 197-280)
@dataclass
class Foundation(IdxBasis):
    """Basic class to manage common attributes and validation logic
    for dimensional analysis entities."""

    _name: str = ""
    description: str = ""

    def __str__(self) -> str:
        _attr_lt = []
        for attr, value in vars(self).items():
            if attr.startswith("__"):
                continue
            if callable(value):
                value = f"{value.__name__}{inspect.signature(value)}"
            _attr_name = attr.lstrip("_")
            _attr_lt.append(f"{_attr_name}={repr(value)}")
        return f"{self.__class__.__name__}({', '.join(_attr_lt)})"
```

### The Inheritance Chain

Every domain object shares this spine:

```
SymBasis (ABC)        — symbol, framework, alias + validated setters
  └── IdxBasis        — index/precedence + validated setter
        └── Foundation  — name, description, __str__/__repr__
              ├── Dimension       (dimensional/fundamental.py)
              ├── Coefficient     (dimensional/buckingham.py)
              ├── Schema          (dimensional/vaschy.py, via IdxBasis)
              └── ConceptualSpecs (elements/specs/conceptual.py)
                    └── Variable  (via 4-spec composition, section 6)
```

Each `clear()` method chains upward through `super().clear()`, resetting every layer's attributes in reverse order. This means calling `variable.clear()` resets statistical, numerical, symbolic, conceptual, AND foundation-level attributes in one call — no attribute gets forgotten because the chain is structural, not manual.

**What this enables downstream:** Any new entity class that needs a symbol, a framework context, and an index simply inherits from `Foundation` and gets all of it — validated setters, consistent `__str__`, and a `clear()` chain — for free. The Schema (section 5), the Dimension, and the Coefficient all share exactly the same symbol-handling logic because they share the same base class, not because someone copy-pasted it.

---

## How These Patterns Connect

The seven patterns form a layered pipeline. Each layer builds on the one below it:

1. **JSON config** (`default.json`) defines the available FDUs and frameworks — the raw data.

2. **The singleton** (`PyDASAConfig`) loads the JSON once, freezes it, and exposes it globally — the data becomes immutable state.

3. **The Foundation layer** (`SymBasis -> IdxBasis -> Foundation`) gives every entity a validated symbol, framework, index, and name — the structural skeleton.

4. **The validation decorators** enforce constraints at every property setter — types, ranges, patterns, choices — using the Schema's regex and the singleton's enum definitions.

5. **The Schema** reads from the singleton, inherits from the Foundation, and generates framework-specific regex and Dimension objects — the dimensional context.

6. **Variables** compose four perspectives on top of the Foundation (conceptual, symbolic, numerical, statistical), each validated by the decorators, and parse their LaTeX symbols through the serialization layer.

7. **The Matrix** collects validated Variables, constructs the dimensional matrix using their Schema-processed `_dim_col` vectors, solves it via RREF, and produces Coefficient objects (which also inherit from Foundation) whose expressions are built from the same LaTeX pipeline.

The result is a data flow where a change to the JSON config propagates through the singleton, into every Schema, through the Foundation's validated setters, and ultimately into the matrix algebra — without touching any code. A change to a validation rule in the decorators applies uniformly across Variables, Coefficients, Dimensions, and Schemas because they all share the same Foundation base class.

This is a library designed for a single maintainer doing research. The patterns are not premature abstractions — they solve real problems: the Foundation ensures every entity validates consistently, the immutable singleton prevents state mutation during long analysis sessions, the decorators prevent silent data corruption, and the compositional Variable design prevents the kind of 2000-line god class that becomes unmaintainable.
