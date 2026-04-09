# Code Highlights: Monte Carlo and Sensitivity Analysis Subsystems

**PyDASA v0.7.0** | 2026-04-08

This report walks through the architectural patterns that make PyDASA's analysis subsystems work. Both the sensitivity analysis and Monte Carlo simulation pipelines share a common structural design -- an orchestrator that manages shared state and delegates the per-coefficient work to individual worker objects. The code is surprisingly consistent across the two subsystems, which makes it easier to reason about but also introduces some duplication worth discussing.

The files covered are:

| File | Role |
|------|------|
| `analysis/scenario.py` | `Sensitivity` worker -- per-coefficient sensitivity analysis |
| `analysis/simulation.py` | `MonteCarlo` worker -- per-coefficient MC simulation |
| `workflows/influence.py` | `SensitivityAnalysis` orchestrator |
| `workflows/practical.py` | `MonteCarloSimulation` orchestrator |
| `workflows/basic.py` | `WorkflowBase` shared base class |
| `core/basic.py` | `Foundation` base class (identity, validation) |

---

## 1. Two-Layer Architecture (Orchestrator + Worker)

The central structural decision in both subsystems is a two-layer split: an orchestrator holds the full set of variables, coefficients, and the dimensional schema, while lightweight worker objects handle the actual computation for one coefficient at a time. This exists because the same physical variable (say, density or velocity) participates in multiple dimensionless coefficients. The orchestrator owns that variable once; each worker receives a reference to it and focuses on parsing and evaluating a single coefficient expression.

Both orchestrators inherit from `Foundation` (which provides identity fields like `name`, `sym`, `idx`, `fwk`) and `WorkflowBase` (which provides `_variables`, `_coefficients`, `_schema`, `_results`, and `_is_solved`). The workers inherit only from `Foundation` and carry their own local copies of the variables they need.

**WorkflowBase provides the shared foundation** (`workflows/basic.py:44-82`):

```python
@dataclass
class WorkflowBase:
    # Common workflow components
    _variables: Dict[str, Variable] = field(default_factory=dict)
    _schema: Optional[Schema] = None
    _coefficients: Dict[str, Coefficient] = field(default_factory=dict)
    _results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _is_solved: bool = False
```

**SensitivityAnalysis creates one Sensitivity worker per coefficient** (`workflows/influence.py:111-134`):

```python
def _create_analyses(self) -> None:
    self._analyses.clear()

    for i, (pi, coef) in enumerate(self._coefficients.items()):
        analysis = Sensitivity(
            _idx=i,
            _sym=f"SEN_{{{coef.sym}}}",
            _fwk=self._fwk,
            _cat=self._cat,
            _name=f"Sensitivity for {coef.name}",
            description=f"Sensitivity analysis for {coef.sym}",
        )
        analysis.set_coefficient(coef)
        self._analyses[pi] = analysis
```

**MonteCarloSimulation does the same, but also wires in distributions and the shared cache** (`workflows/practical.py:277-351`):

```python
def configure_simulations(self) -> None:
    self._simulations.clear()

    if not self._shared_cache:
        self._init_shared_cache()

    for i, (pi, coef) in enumerate(self._coefficients.items()):
        var_dims = self._validate_coefficient_vars(coef, pi)
        vars_in_coef = list(var_dims.keys())

        sim = MonteCarlo(
            _idx=i,
            _sym=f"MC_{{{coef.sym}}}",
            _fwk=self._fwk,
            _cat=self._cat,
            _pi_expr=coef.pi_expr,
            _coefficient=coef,
            _variables=self._variables,
            _experiments=self._experiments,
            _name=f"Monte Carlo Simulation for {coef.name}",
            description=f"Monte Carlo simulation for {coef.sym}",
        )
        sim.set_coefficient(coef)
        sim._distributions = self._get_distributions(vars_in_coef)
        sim._dependencies = self._get_dependencies(vars_in_coef)
        # CRITICAL: Share the cache reference
        sim._simul_cache = self._shared_cache
        self._simulations[pi] = sim
```

This pattern enables clean separation: the orchestrator handles configuration, validation, and result consolidation; the worker handles parsing, sampling, and evaluation. Adding a new coefficient to the analysis just means creating another worker -- no changes to the orchestrator logic.

---

## 2. Dual-Mode Sensitivity (Symbolic + Numerical)

The `Sensitivity` class offers two fundamentally different ways to analyze the same coefficient expression. Symbolic mode uses SymPy's exact differentiation to compute partial derivatives at specific variable values -- this tells you the local rate of change of the coefficient with respect to each variable. Numerical mode uses SALib's FAST (Fourier Amplitude Sensitivity Test) to compute global sensitivity indices across a range of variable values -- this tells you what fraction of the output variance each variable is responsible for. Both modes start from the same parsed LaTeX expression, which makes them directly comparable.

**Symbolic mode: differentiate, compile, evaluate** (`analysis/scenario.py:304-348`):

```python
def analyze_symbolically(self,
                         vals: Dict[str, float]) -> Dict[str, float]:
    self._validate_analysis_ready()
    self.var_values = vals

    var_lt = [str(v) for v in self._latex_to_py]
    missing_vars = set(var_lt) - set(list(vals.keys()))
    if missing_vars:
        raise ValueError(f"Missing values for variables: {missing_vars}")

    py_to_latex = self._py_to_latex
    results = dict()
    functions = dict()
    if self._var_names:
        for var in self._var_names:
            expr = diff(self._sym_func, var)
            aliases = [self._aliases[v] for v in self._var_names]
            func = lambdify(aliases, expr, "numpy")
            functions[py_to_latex[var]] = func

            val_args = [vals[py_to_latex[v]] for v in self._var_names]
            res = functions[py_to_latex[var]](*val_args)
            results[py_to_latex[var]] = res

    self._exe_func = functions
    self._results = results
    return self._results
```

The key pipeline here is `diff()` to get the partial derivative as a SymPy expression, then `lambdify()` to compile it into a NumPy-callable function, then evaluation with the user-provided values. The result is one sensitivity value per variable -- the partial derivative evaluated at the given point.

**Numerical mode: FAST sampling and analysis** (`analysis/scenario.py:355-434`):

```python
def analyze_numerically(self,
                        vals: List[str],
                        bounds: List[List[float]],
                        iters: int = -1) -> Dict[str, Any]:
    self._validate_analysis_ready()

    self.experiments = iters
    self.var_bounds = bounds

    if self._var_names:
        problem = {
            "num_vars": len(vals),
            "names": self._var_names,
            "bounds": bounds,
        }
        self._var_domains = sample(problem, self._experiments)
        _len = len(self._var_names)
        self._var_domains = self._var_domains.reshape(-1, _len)

        aliases = [self._aliases[v] for v in self._var_names]
        func = lambdify(aliases, self._sym_func, "numpy")
        self._exe_func = func

        exe_func = self._exe_func
        Y = np.apply_along_axis(lambda v: exe_func(*v),
                                1,
                                self._var_domains)
        self._var_ranges = Y.reshape(-1, 1)

        results = analyze(problem, Y)
```

This mode defines a SALib problem dictionary with variable names and bounds, generates a FAST sample matrix, evaluates the expression across all samples using `np.apply_along_axis`, and then feeds the output vector into SALib's `analyze()` to get first-order and total-order sensitivity indices.

The orchestrator `SensitivityAnalysis` exposes both modes through `analyze_symbolic()` and `analyze_numeric()` methods (`workflows/influence.py:202-282`), which iterate over all workers, gather variable values or bounds from the shared `_variables` dictionary, and call the appropriate worker method.

---

## 3. Shared Cache Architecture (Monte Carlo)

This is the most architecturally significant pattern in the analysis subsystems. When running Monte Carlo simulations across multiple coefficients, the same physical variable appears in different Pi groups. If each coefficient sampled that variable independently, you would get inconsistent scenarios -- variable X might be 5.3 in one coefficient's iteration and 7.1 in another's, within the same "experiment." The shared cache ensures that once a variable is sampled for iteration *i*, every coefficient that needs that variable for iteration *i* gets the same value.

**The orchestrator initializes the cache once for all variables** (`workflows/practical.py:260-275`):

```python
def _init_shared_cache(self) -> None:
    if self._experiments < 0:
        raise ValueError(
            f"Cannot initialize shared cache: experiments must be positive. "
            f"Got: {self._experiments}"
        )

    for var_sym in self._variables.keys():
        self._shared_cache[var_sym] = np.full(
            (self._experiments, 1),
            np.nan,
            dtype=np.float64
        )
```

The cache is a dictionary mapping variable symbols to `(N, 1)` NumPy arrays filled with NaN. NaN serves as the "not yet sampled" sentinel.

**Each worker receives a reference to the same cache object** (`workflows/practical.py:344`):

```python
sim._simul_cache = self._shared_cache
```

This is a reference assignment, not a copy. Every `MonteCarlo` worker's `_simul_cache` attribute points to the same dictionary in memory.

**Workers read and write through accessor methods** (`analysis/simulation.py:828-878`):

```python
def _get_cached_value(self, var_sym: str, idx: int) -> Optional[float]:
    cache_data = None
    if self._validate_cache_locations(var_sym, idx):
        cache_data = self._simul_cache[var_sym][idx, 0]
        if not np.isnan(cache_data):
            cache_data = float(cache_data)
    return cache_data

def _set_cached_value(self,
                      var_sym: str,
                      idx: int,
                      val: Union[float, Dict]) -> None:
    cache_updates = val if isinstance(val, dict) else {var_sym: val}
    if not self._validate_cache_locations(list(cache_updates.keys()), idx):
        invalid_vars = list(cache_updates.keys())
        raise ValueError(
            f"Invalid cache location at index {idx}. "
            f"For variables: {invalid_vars}"
        )
    for k, v in cache_updates.items():
        self._simul_cache[k][idx, 0] = v
```

**The sampling loop checks the cache before generating** (`analysis/simulation.py:457-482`):

```python
for i in range(self._experiments):
    memory: Dict[str, float] = {}
    for var in vars.values():
        cached_val = self._get_cached_value(var.sym, i)
        if cached_val is None or np.isnan(cached_val):
            val = self._generate_sample(var, memory)
            memory[var.sym] = val
            self._set_cached_value(var.sym, i, val)
        else:
            memory[var.sym] = cached_val
        _dataset[var.sym][i] = memory[var.sym]
```

The first coefficient to run samples variable X at iteration *i* and writes it to the cache. The second coefficient finds that value already cached and uses it directly. This guarantees physical consistency across the entire simulation.

---

## 4. Dependency-Aware Sampling

Some variables in a dimensional analysis are not independent -- for example, force depends on mass and acceleration (F = m * a). When sampling, you cannot draw F from its own distribution and then separately draw m and a, because the three values would be physically inconsistent. The Monte Carlo worker handles this through a per-iteration `memory` dictionary and a dependency-aware sampling function.

**The `_generate_sample` method resolves dependencies before sampling** (`analysis/simulation.py:550-605`):

```python
def _generate_sample(self,
                     var: Variable,
                     memory: Dict[str, float]) -> float:
    data: float = -1.0
    _type = (list, tuple, np.ndarray)

    # Get dependency values from memory
    chace_deps = []
    for dep in var.depends:
        if dep in memory:
            dep_val = memory[dep]
            if isinstance(dep_val, (list, tuple, np.ndarray)):
                dep_val = dep_val[-1]
            chace_deps.append(dep_val)

    if var._dist_func is not None:
        # Independent variable
        if not var.depends:
            data = var.sample()
        # Dependent variable with all deps resolved
        elif len(var.depends) == len(chace_deps):
            raw_data = var.sample(*chace_deps)
            if isinstance(raw_data, _type):
                data = raw_data[-1]
                for dep in var.depends:
                    if dep in memory:
                        memory[dep] = raw_data[var.depends.index(dep)]
            else:
                data = raw_data

    memory[var.sym] = float(data)
    return data
```

The method first looks up any dependency values from the current iteration's `memory` dict. If the variable is independent, it calls `var.sample()` with no arguments. If it has dependencies and all of them have been sampled already, it passes those values into `var.sample(*chace_deps)`, which invokes the variable's distribution function with the dependency values as arguments.

This is driven by two attributes on `Variable`: `_depends` (a list of variable symbols this variable depends on) and `_dist_func` (the sampling callable). The iteration loop in `_generate_dataset` processes variables in order, building up the `memory` dict so that by the time a dependent variable is reached, its dependencies are already resolved.

---

## 5. Dual Data Modes (DIST vs DATA)

The `MonteCarlo` worker can operate in two modes: `DIST` (sample from distribution functions) or `DATA` (use pre-existing datasets attached to variables). This makes the same pipeline suitable for both stochastic forward analysis and replay of historical or experimental data.

**The `run()` method switches on mode** (`analysis/simulation.py:607-656`):

```python
def run(self,
        iters: Optional[int] = None,
        mode: str = SimulationMode.DIST.value) -> None:
    self.cat = mode
    if iters is not None:
        self.experiments = iters
    self._validate_readiness()
    self._reset_memory()

    aliases = [self._aliases[v] for v in self._var_symbols]
    self._exe_func = lambdify(aliases, self._sym_func, "numpy")

    vars_in_expr = {}
    for k, v in self._variables.items():
        if k in self._latex_to_py or k in self._var_symbols or v._alias in self._var_symbols:
            vars_in_expr[k] = v

    if self._cat == SimulationMode.DATA.value:
        self._data = self._collect_dataset(vars_in_expr)
    elif self._cat == SimulationMode.DIST.value:
        self._data = self._generate_dataset(vars_in_expr)
```

**`_collect_dataset` validates consistency across variables** (`analysis/simulation.py:400-434`):

```python
def _collect_dataset(self,
                     vars: Dict[str, Variable]) -> Dict[str, NDArray[np.float64]]:
    _dataset = {}
    _exp_len = None

    for sym, var in vars.items():
        if var.data is None or len(var.data) == 0:
            raise ValueError(f"Variable '{var.sym}' has no data. ")
        _cur_len = len(var.data)
        if _exp_len is None:
            _exp_len = _cur_len
        elif _cur_len != _exp_len:
            raise ValueError(
                f"Variable '{var.sym}' has {_cur_len} data points, "
                f"but expected {_exp_len}."
            )
        _dataset[sym] = np.array(var.data, dtype=np.float64)
    return _dataset
```

After either mode produces the `_data` dictionary, the evaluation loop at `simulation.py:659-696` is identical -- it iterates over experiments, gathers variable values for each iteration from `_data`, orders them according to `_var_symbols`, and evaluates the compiled expression. This means the downstream statistics (mean, median, confidence intervals) work the same regardless of where the data came from.

---

## 6. Expression Pipeline (LaTeX to SymPy to NumPy)

Both `Sensitivity` and `MonteCarlo` need to turn a LaTeX coefficient expression (like `\frac{\rho \cdot v^2}{P}`) into something they can evaluate numerically. The pipeline has three stages: parse the LaTeX into a SymPy expression, substitute LaTeX-style symbols with Python-friendly symbols, and compile the result into a NumPy-callable function via `lambdify`.

**`_parse_expression` in `Sensitivity`** (`analysis/scenario.py:239-302`):

```python
def _parse_expression(self, expr: str) -> None:
    parsed_expr = parse_latex(expr)

    if parsed_expr is None:
        raise ValueError("parse_latex returned 'None'.")
    if not isinstance(parsed_expr, sp.Expr):
        raise TypeError(
            f"Parsed expression is not a SymPy expression: "
            f"{type(parsed_expr).__name__}."
        )

    maps = create_latex_mapping(expr)
    self._symbols = maps[0]
    self._aliases = maps[1]
    self._latex_to_py = maps[2]
    self._py_to_latex = maps[3]

    sym_func: sp.Expr = parsed_expr
    for latex_sym, py_sym in self._symbols.items():
        result = sym_func.subs(latex_sym, py_sym)
        if isinstance(result, sp.Expr):
            sym_func = result
        else:
            sym_func = sp.sympify(result)

    self._sym_func = sym_func
    fsyms = sym_func.free_symbols
    self._var_names = sorted([str(s) for s in fsyms])
```

**`_parse_expression` in `MonteCarlo`** (`analysis/simulation.py:484-544`):

```python
def _parse_expression(self, expr: str) -> None:
    self._sym_func = parse_latex(expr)
    if self._sym_func is None:
        raise ValueError("Parsing returned None")

    maps = create_latex_mapping(expr)
    symbols_raw: Dict[Any, sp.Symbol] = maps[0]
    aliases_raw: Dict[str, sp.Symbol] = maps[1]
    latex_to_py: Dict[str, str] = maps[2]
    py_to_latex: Dict[str, str] = maps[3]

    self._symbols = {str(k): v for k, v in symbols_raw.items()}
    self._aliases = aliases_raw
    self._latex_to_py = latex_to_py
    self._py_to_latex = py_to_latex

    for latex_sym_key, py_sym in symbols_raw.items():
        if isinstance(latex_sym_key, sp.Symbol):
            self._sym_func = self._sym_func.subs(latex_sym_key, py_sym)
        else:
            latex_symbol = sp.Symbol(str(latex_sym_key))
            self._sym_func = self._sym_func.subs(latex_symbol, py_sym)

    if self._sym_func is not None and hasattr(self._sym_func, "free_symbols"):
        free_symbols = self._sym_func.free_symbols
        self._var_symbols = sorted([str(s) for s in free_symbols])
```

Both methods call `parse_latex()` and `create_latex_mapping()` from `pydasa.serialization.parser`, then substitute symbols and extract the free symbol list. The critical detail is the ordering: `_var_names` (in `Sensitivity`) and `_var_symbols` (in `MonteCarlo`) are both `sorted()`, which ensures that when `lambdify` compiles the expression, the argument order is deterministic and matches the order used when passing values during evaluation.

The compilation step happens later -- in `analyze_symbolically` (per derivative), in `analyze_numerically` (for the whole expression), and in `run()` for Monte Carlo. In all cases the pattern is the same:

```python
aliases = [self._aliases[v] for v in self._var_symbols]
self._exe_func = lambdify(aliases, self._sym_func, "numpy")
```

This produces a Python function that accepts NumPy scalars or arrays and returns the evaluated expression.

---

## 7. How They Connect

The full pipeline from raw problem definition to analysis results follows a consistent path through these layers:

1. **AnalysisEngine** (in `workflows/phenomena.py`) takes a set of `Variable` objects and a dimensional `Schema`, builds a dimensional matrix, and produces `Coefficient` objects. Each coefficient carries a LaTeX expression (`_pi_expr`) and a record of which variables participate (`var_dims`).

2. **SensitivityAnalysis** or **MonteCarloSimulation** receives those coefficients plus the original variables via the `WorkflowBase` interface. The orchestrator stores them in `_coefficients` and `_variables`.

3. The orchestrator creates one worker per coefficient. For sensitivity analysis, `_create_analyses()` instantiates `Sensitivity` objects and calls `set_coefficient()`. For Monte Carlo, `configure_simulations()` instantiates `MonteCarlo` objects, wires in distributions and dependencies, and assigns the shared cache reference.

4. Each worker parses the coefficient's LaTeX expression into a SymPy function, builds the symbol mappings, and compiles an executable function via `lambdify`.

5. **Sensitivity** either differentiates the expression symbolically (producing one derivative per variable) or generates FAST samples for global sensitivity indices. **MonteCarlo** iterates through experiments, checking the shared cache before sampling each variable, then evaluates the compiled function with the sampled values.

6. Results flow back to the orchestrator. `SensitivityAnalysis` stores them in `_results` keyed by coefficient symbol. `MonteCarloSimulation` stores inputs, raw results, and statistics for each coefficient.

The shared cache is what ties the Monte Carlo workers together -- without it, each coefficient's simulation would be an independent experiment, and you could not meaningfully compare results across coefficients for the same iteration.

---

## 8. Proposals

**Unify the two `_parse_expression` methods.** The `Sensitivity._parse_expression` and `MonteCarlo._parse_expression` do the same thing with minor differences (one checks `isinstance(parsed_expr, sp.Expr)`, the other does not; one stores symbols as SymPy objects, the other converts keys to strings). A shared mixin or utility function in `serialization/parser.py` -- something like `parse_coefficient_expression(expr) -> (sym_func, symbols, aliases, latex_to_py, py_to_latex, var_names)` -- would eliminate 40-50 lines of near-duplicate code and ensure both subsystems stay in sync when the parser changes.

**Add Sobol indices alongside FAST.** The numerical sensitivity analysis currently only supports FAST. Sobol indices (via SALib's `saltelli` sampler and `sobol.analyze`) provide a more widely recognized variance-based decomposition and give second-order interaction indices that FAST does not. Since the infrastructure is already there -- `problem` dict, `lambdify`-compiled function, `np.apply_along_axis` evaluation -- adding Sobol would be a small extension to `analyze_numerically` or a new method on `Sensitivity`. This could be a `lab/` PoC first.

**Introduce convergence diagnostics for Monte Carlo.** The current MC pipeline runs a fixed number of iterations and computes summary statistics. There is no mechanism to detect whether the results have converged -- whether running 10,000 more iterations would materially change the mean or standard deviation. A simple running-mean convergence check (compute the mean after every N iterations and stop when the relative change drops below a threshold) would make the simulation more robust and could save significant computation time for well-behaved distributions.

**Formalize the shared cache as a first-class object.** Right now the shared cache is a plain dictionary created by `_init_shared_cache()` and passed by reference. This works, but it means any code with a reference can write to any variable at any index without validation (the validation is in the worker, not the cache itself). Wrapping it in a small `SimulationCache` class with explicit `get(var, idx)` and `set(var, idx, value)` methods -- plus a `freeze()` method to prevent further writes after simulation completes -- would make the protocol explicit and easier to debug.

**Consider vectorizing the Monte Carlo sampling loop.** The current `_generate_dataset` method loops over iterations one at a time (`for i in range(self._experiments)`), calling `_generate_sample` per variable per iteration. For independent variables with standard distributions (normal, uniform, etc.), this could be replaced with a single vectorized call like `np.random.normal(mu, sigma, size=N)`, which would be orders of magnitude faster. The per-iteration loop would only be needed for variables with dependencies. A hybrid approach -- vectorize independent variables, loop only for dependent ones -- could give a substantial speedup on large experiment counts.

**Add a dry-run or validation mode for MonteCarloSimulation.** Before running thousands of iterations, it would be useful to have a method that runs 5-10 iterations and reports whether all variables sampled successfully, all cache lookups resolved, and all coefficient evaluations produced finite values. This would catch configuration errors (missing distributions, broken dependency chains, expression parsing issues) early, before committing to a long simulation run.
