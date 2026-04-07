# Code Documentation Skill

This skill produces accurate, well-structured documentation from source code. It covers four output types: inline comments, docstrings, module/package READMEs, and API reference sections. Each has its own workflow below.

---

## Step 0 — Read the Code First

Before writing anything, always read and understand the code. If files are uploaded or referenced, use the appropriate tool to load them. If a directory is provided, scan the structure first:

```bash
find <dir> -name "*.py" | head -40
```

Build a mental model of:
- **Purpose**: what does this code do at a high level?
- **Structure**: modules, classes, functions and their relationships
- **Public API**: what is meant to be used externally vs. internal implementation
- **Patterns**: design patterns, error handling conventions, data flow
- **Dependencies**: what does this code rely on?

Do not start writing documentation until this model is clear. If the code is ambiguous, ask the user before guessing.

---

## Output Type 1 — Inline Comments

Use when: code logic is non-obvious, algorithm steps need labelling, or magic values need explaining.

**Rules:**
- Comment the *why*, not the *what*. `# increment counter` is noise. `# retry limit reached; fall back to default` is useful.
- Place comments on the line before the code they describe, not after.
- Keep comments short — one line where possible.
- Do not comment every line. Sparse, precise comments are better than dense ones.
- Flag TODOs or known limitations with `# TODO:` or `# NOTE:`.

**Output format:** Return the annotated file with `str_replace` if editing an existing file, or as a code block if producing new output.

---

## Output Type 2 — Docstrings

Choose the style that matches the project's existing convention. If there is no convention, default to **Google style** for research/scientific Python, **NumPy style** for numerical/scientific libraries, or **Sphinx/reST** for projects generating API docs with Sphinx.

### Google Style (default for research code)

```python
def function(arg1: type, arg2: type) -> return_type:
    """Short one-line summary.

    Longer description if needed. Explain behaviour, not implementation.
    Mention edge cases, preconditions, or constraints here.

    Args:
        arg1: Description of arg1. Include units if numeric.
        arg2: Description of arg2.

    Returns:
        Description of return value and its type.

    Raises:
        ValueError: When and why this is raised.
        TypeError: When and why this is raised.

    Example:
        >>> result = function(x, y)
        >>> print(result)
        expected_output
    """
```

### NumPy Style (for scientific/numerical libraries)

```python
def function(arg1, arg2):
    """
    Short one-line summary.

    Extended description if needed.

    Parameters
    ----------
    arg1 : type
        Description. Include units if numeric (e.g., kg/m^3).
    arg2 : type
        Description.

    Returns
    -------
    type
        Description of return value.

    Raises
    ------
    ValueError
        When and why.

    Examples
    --------
    >>> result = function(x, y)
    expected_output
    """
```

### Class Docstrings

Document the class in the class body docstring, not `__init__`. The `__init__` docstring should document parameters only if `__init__` has significant logic beyond assignment.

```python
class MyClass:
    """One-line summary of the class.

    Longer description of the class's role, responsibility, and key invariants.
    Describe what the class represents, not how it is implemented.

    Attributes:
        attr1: Description of public attribute.
        attr2: Description of public attribute.

    Example:
        >>> obj = MyClass(x, y)
        >>> obj.method()
    """
```

### Module Docstrings

Every module should have a top-level docstring:

```python
"""
Module name and one-line purpose.

Extended description of the module's role within the package.
List key public classes and functions if the module is large.

Typical usage::

    from package.module import ClassName
    obj = ClassName(args)
    result = obj.method()
"""
```

---

## Output Type 3 — README / Package-Level Documentation

Use when the user wants a written description of a library, package, or repository for human readers (not just API consumers).

### Standard README Structure

Adapt sections to what is present in the code. Do not include sections that would be empty.

```markdown
# Package Name

One-sentence description.

## Overview

2-4 paragraph description covering:
- What problem this solves
- Who it is for
- What makes it distinct from alternatives (if known)

## Installation

pip / conda / from source instructions.

## Quick Start

Minimal working example. Choose an example where the expected output
is verifiable (e.g., a known mathematical result). Annotate each step.

## Core Concepts

Brief explanation of the key abstractions (classes, objects, frameworks).
Reference the API section for details.

## Architecture (optional)

High-level description of the package structure and module responsibilities.
Include the directory tree if helpful.

## API Reference

See Output Type 4 below, or link to generated docs.

## Limitations

Honest statement of what the package does not yet support.

## Licence and Citation

Licence statement. Citation block if applicable.
```

**Voice guidance:**
- Use first person plural ("we") for the author's own decisions and design choices.
- Use third person for external tools and dependencies ("NumPy provides...").
- Keep sentences short and declarative. Avoid marketing language.
- Do not pad with filler. Every sentence should carry information.

---

## Output Type 4 — API Reference Section

Use when: producing a reference section for a dissertation chapter, technical report, or documentation site that describes the public API systematically.

### Structure per module

For each public module, produce:

1. **Module summary** — one paragraph stating the module's role within the package.
2. **Public classes** — for each class: purpose, key attributes (tabular), key methods (tabular with signature and brief description).
3. **Public functions** — for each function: signature, one-line description, parameter table, return type.
4. **Usage example** — a short code snippet demonstrating the most common use case.

### Parameter table format

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Symbolic name used as dictionary key and LaTeX symbol |
| `dims` | `str` | Dimensional formula, e.g. `"M*L^-3"` |

### Method table format

| Method | Signature | Description |
|--------|-----------|-------------|
| `to_dict()` | `() -> dict` | Exports all attributes as a plain Python dictionary |
| `calculate_setpoint()` | `() -> float` | Evaluates the coefficient numerically from stored setpoints |

---

## Decision Guide — Which Output Type?

| User says... | Output type |
|---|---|
| "Add comments to this" | Inline comments |
| "Document this function / class" | Docstrings |
| "Write docstrings for the whole file" | Docstrings (all public members) |
| "Write a README for this library" | README |
| "Describe this library for my dissertation" | API Reference + README sections |
| "What does this code do?" | Read and summarise, then ask which output type |
| "Document this for publication / paper" | README + API Reference |

If the user's intent is ambiguous, ask:
> "Do you want inline comments in the code itself, docstrings on functions and classes, a README for the repository, or an API reference section for a report?"

---

## Quality Checks Before Delivering

Before returning any documentation output, verify:

- [ ] Every public function, class, and module has a docstring
- [ ] All parameters, return types, and raised exceptions are documented
- [ ] At least one usage example is included per class or module
- [ ] No internal implementation detail is exposed in the public API docs
- [ ] Units are specified for all numeric parameters where applicable
- [ ] The style is consistent throughout (Google vs NumPy vs Sphinx)
- [ ] Docstrings describe *what* and *why*, not *how*
- [ ] The language matches the project's existing style

---

## Docstring Reference Examples

### Abstract Base Class

```python
from abc import ABC, abstractmethod

class BaseWorkflow(ABC):
    """Abstract base class for all dimensional analysis workflow classes.

    Subclasses must implement run() and must accept Variable objects
    as input and produce Coefficient objects as output. This contract
    is enforced by the validations layer at instantiation time.

    Attributes:
        _idx: Unique integer identifier for this workflow instance.
        variables: Dict mapping symbolic names to Variable objects.
        coefficients: Dict mapping Pi group labels to Coefficient objects,
            populated after run() is called.
    """

    @abstractmethod
    def run(self) -> dict:
        """Execute the workflow and return results.

        Returns:
            Dict mapping coefficient labels to Coefficient objects.

        Raises:
            ValidationError: If variables are not set before calling run().
        """
```

### Decorated Function

```python
def validate_variables(func):
    """Decorator that validates the variable set before executing func.

    Checks that exactly one variable has _cat='OUT', that at least one
    variable has relevant=True, and that all _dims strings are parseable.
    Raises ValidationError with a targeted message on any violation.

    Args:
        func: The method to wrap. Expected to be a method of AnalysisEngine
            or a subclass thereof.

    Returns:
        Wrapped function that validates variables before delegation to func.

    Raises:
        ValidationError: If validation fails. The message identifies the
            specific variable and attribute that caused the failure.

    Example:
        @validate_variables
        def run_analysis(self):
            ...
    """
```

### Function with Optional and Complex Parameters

```python
def derive_coefficient(
    self,
    expr: str,
    symbol: str,
    name: str,
    setpoint_overrides: dict | None = None,
) -> "Coefficient":
    """Derive a new dimensionless coefficient from existing primary groups.

    Creates a Coefficient by evaluating a symbolic expression over the
    engine's existing coefficients. The derived coefficient inherits the
    constituent variables of all coefficients referenced in expr.

    Args:
        expr: Symbolic expression string using existing coefficient labels
            as variables. For example, ``"1/Pi_0"`` or ``"Pi_0 * Pi_1"``.
            Standard Python arithmetic operators are supported.
        symbol: Short symbol for the new coefficient (e.g., ``"Re"``).
            Used in LaTeX rendering and plot axis labels.
        name: Human-readable name (e.g., ``"Reynolds Number"``).
            Used in report generation and DataFrame column headers.
        setpoint_overrides: Optional dict mapping variable symbolic names
            to float values. When provided, overrides the _std_setpoint
            stored on the constituent Variable objects for this evaluation
            only. Useful for parametric sensitivity checks without
            modifying the variable definitions.

    Returns:
        A Coefficient object with pi_expr set to the evaluated symbolic
        expression and calculate_setpoint() bound to the constituent
        variables' setpoints (or overrides if provided).

    Raises:
        KeyError: If expr references a coefficient label that does not
            exist in engine.coefficients.
        ValueError: If expr is not a valid symbolic expression.

    Example:
        Deriving the Reynolds number as the reciprocal of the primary Pi group::

            pi_0_key = list(engine.coefficients.keys())[0]
            Re = engine.derive_coefficient(
                expr=f"1/{pi_0_key}",
                symbol="Re",
                name="Reynolds Number",
            )
            print(Re.calculate_setpoint())   # 100000.0
    """
```

### Property

```python
@property
def variables(self) -> dict:
    """The engine's variable dictionary, keyed by symbolic name.

    Each value is a Variable object. Setting this property triggers
    validation of the full variable set via the validations layer.

    Setting with plain dicts is supported; they are normalised to
    Variable objects transparently.

    Returns:
        Dict mapping symbolic name strings to Variable objects.

    Raises:
        ValidationError: If the assigned value fails structural validation
            (e.g., missing _cat, unparseable _dims, zero relevant variables).

    Example:
        >>> engine.variables = {"x": {"_cat": "IN", ...}}
        >>> type(engine.variables["x"])
        <class 'pydasa.elements.parameter.Variable'>
    """
```

### Style Consistency Checklist

When documenting a full module, ensure:

- All docstrings use the same style (Google / NumPy / Sphinx)
- Units appear in Args/Parameters for every numeric parameter
- All `Raises` blocks are complete — every exception the function can raise is listed
- Private methods (prefixed `_`) do not require public docstrings unless they are called from tests or subclasses
- `__repr__` and `__str__` should have brief one-line docstrings
- `__init__` parameters are documented on the class docstring (Google) or in the Parameters section of the class docstring (NumPy), not separately on `__init__`
