# Migration Guide: Using PyDASAConfig Singleton and Enums

This guide shows how to migrate from the deprecated dictionary-based configuration (`FDU_FWK_DT`, `PARAMS_CAT_DT`, etc.) to the new type-safe singleton pattern using `PyDASAConfig` and Enum classes.

## Overview

The new approach provides:
- **Type safety** through Enums instead of string dictionaries
- **Singleton pattern** for consistent configuration access
- **Better IDE support** with autocomplete and type checking
- **Immutability** through frozen dataclass

## Quick Reference

### Old (Deprecated) ❌
```python
from pydasa.core.setup import FDU_FWK_DT, PARAMS_CAT_DT

# Using string keys with dictionary
if framework in FDU_FWK_DT:
    description = FDU_FWK_DT[framework]

# Validation with dictionary
@validate_choices(FDU_FWK_DT)
def fwk(self, val: str) -> None:
    self._fwk = val
```

### New (Recommended) ✅
```python
from pydasa.core.setup import PyDASAConfig, Framework, VarCardinality

# Using Enums directly
config = PyDASAConfig.get_instance()
if framework in [f.value for f in Framework]:
    description = Framework[framework].description

# Validation with Enum values
@validate_choices([f.value for f in Framework])
def fwk(self, val: str) -> None:
    self._fwk = val
```

## Available Enums

### 1. Framework
Fundamental Dimensional Units (FDUs) frameworks.

```python
from pydasa.core.setup import Framework

# Available frameworks
Framework.PHYSICAL      # "PHYSICAL" - Traditional physical dimensions
Framework.COMPUTATION   # "COMPUTATION" - Computer science dimensions
Framework.SOFTWARE      # "SOFTWARE" - Software architecture dimensions
Framework.CUSTOM        # "CUSTOM" - User-defined dimensions

# Access enum value
framework_name = Framework.PHYSICAL.value  # Returns "PHYSICAL"

# Access description
description = Framework.PHYSICAL.description
# Returns: "Traditional physical dimensional framework (e.g., Length, Mass, Time)."

# Get all frameworks
all_frameworks = list(Framework)
framework_values = [f.value for f in Framework]
```

### 2. VarCardinality
Variable cardinality categories.

```python
from pydasa.core.setup import VarCardinality

# Available cardinalities
VarCardinality.IN       # "IN" - Input variables
VarCardinality.OUT      # "OUT" - Output variables
VarCardinality.CTRL     # "CTRL" - Control variables

# Access enum value and description
cardinality = VarCardinality.IN.value  # "IN"
description = VarCardinality.IN.description
# Returns: "Variables that influence the system (e.g., known inputs)."
```

### 3. CoefCardinality
Dimensionless Coefficient cardinality.

```python
from pydasa.core.setup import CoefCardinality

# Available cardinalities
CoefCardinality.COMPUTED    # "COMPUTED" - Directly calculated
CoefCardinality.DERIVED     # "DERIVED" - Combined/manipulated

# Usage
coef_type = CoefCardinality.COMPUTED.value
description = CoefCardinality.COMPUTED.description
# Returns: "Coefficients directly calculated using the Dimensional Matrix."
```

### 4. AnaliticMode
Analysis modes for dimensional analysis.

```python
from pydasa.core.setup import AnaliticMode

# Available modes
AnaliticMode.SYM    # "SYM" - Symbolic analysis
AnaliticMode.NUM    # "NUM" - Numeric analysis

# Usage
mode = AnaliticMode.SYM.value
description = AnaliticMode.SYM.description
# Returns: "Analysis for symbolic processable parameters (e.g., 'z = x + y')."
```

## Using the Singleton Pattern

### Basic Usage

```python
from pydasa.core.setup import PyDASAConfig

# Get the singleton instance (creates if doesn't exist)
config = PyDASAConfig.get_instance()

# Access all frameworks
all_frameworks = config.frameworks  # Returns tuple of Framework enums
print(all_frameworks)  # (<Framework.PHYSICAL: 'PHYSICAL'>, ...)

# Access all variable cardinalities
all_var_cards = config.parameter_cardinality  # Returns tuple of VarCardinality enums

# Access all coefficient cardinalities
all_coef_cards = config.coefficient_cardinality  # Returns tuple of CoefCardinality enums

# Access all analytic modes
all_modes = config.analitic_modes  # Returns tuple of AnaliticMode enums
```

### In Validation Decorators

```python
from pydasa.validations.decorators import validate_choices
from pydasa.core.setup import Framework, VarCardinality

class MyClass:
    def __init__(self):
        self._framework = Framework.PHYSICAL.value
        self._cardinality = VarCardinality.IN.value
    
    @property
    def framework(self) -> str:
        return self._framework
    
    @framework.setter
    @validate_choices([f.value for f in Framework])
    def framework(self, val: str) -> None:
        """Set framework with enum-based validation."""
        self._framework = val
    
    @property
    def cardinality(self) -> str:
        return self._cardinality
    
    @cardinality.setter
    @validate_choices([c.value for c in VarCardinality])
    def cardinality(self, val: str) -> None:
        """Set cardinality with enum-based validation."""
        self._cardinality = val
```

### Type-Safe Comparisons

```python
from pydasa.core.setup import Framework, VarCardinality

# Compare using enum values (recommended)
if my_framework == Framework.PHYSICAL.value:
    print("Using physical framework")

# Check if value is valid
valid_frameworks = {f.value for f in Framework}
if user_input in valid_frameworks:
    print(f"{user_input} is a valid framework")

# Get enum from string value
try:
    framework_enum = Framework(user_input)
    print(f"Description: {framework_enum.description}")
except ValueError:
    print(f"Invalid framework: {user_input}")
```

## Migration Checklist

When migrating existing code:

1. **Replace dictionary imports:**
   ```python
   # Old
   from pydasa.core.setup import FDU_FWK_DT, PARAMS_CAT_DT
   
   # New
   from pydasa.core.setup import Framework, VarCardinality
   ```

2. **Update validation decorators:**
   ```python
   # Old
   @validate_choices(FDU_FWK_DT)
   
   # New
   @validate_choices([f.value for f in Framework])
   ```

3. **Replace dictionary lookups:**
   ```python
   # Old
   description = FDU_FWK_DT[framework]
   
   # New
   description = Framework[framework].description
   # Or safer with try/except:
   try:
       description = Framework(framework).description
   except ValueError:
       description = "Unknown framework"
   ```

4. **Update default values:**
   ```python
   # Old
   _fwk: str = "PHYSICAL"
   
   # New
   _fwk: str = Framework.PHYSICAL.value
   ```

5. **Update tests:**
   ```python
   # Old
   assert framework in FDU_FWK_DT
   
   # New
   assert framework in [f.value for f in Framework]
   ```

## Benefits of the New Approach

1. **Type Safety:** IDEs can catch typos at development time
2. **Autocomplete:** Get suggestions for valid values
3. **Documentation:** Each enum has a description property
4. **Consistency:** Singleton ensures same config everywhere
5. **Maintainability:** Central enum definitions
6. **Extensibility:** Easy to add new values
7. **Immutability:** Frozen dataclass prevents accidental modification

## Example: Complete Class Migration

### Before (Old Dictionary-Based)

```python
from pydasa.core.setup import FDU_FWK_DT
from pydasa.validations.decorators import validate_choices

class OldVariable:
    def __init__(self):
        self._framework = "PHYSICAL"
    
    @property
    def framework(self) -> str:
        return self._framework
    
    @framework.setter
    @validate_choices(FDU_FWK_DT)
    def framework(self, val: str) -> None:
        self._framework = val
    
    def is_valid_framework(self, fwk: str) -> bool:
        return fwk in FDU_FWK_DT
```

### After (New Enum-Based)

```python
from pydasa.core.setup import PyDASAConfig, Framework
from pydasa.validations.decorators import validate_choices

class NewVariable:
    # Access singleton once at class level (optional, for efficiency)
    _config = PyDASAConfig.get_instance()
    
    def __init__(self):
        self._framework = Framework.PHYSICAL.value
    
    @property
    def framework(self) -> str:
        return self._framework
    
    @framework.setter
    @validate_choices([f.value for f in Framework])
    def framework(self, val: str) -> None:
        self._framework = val
    
    def is_valid_framework(self, fwk: str) -> bool:
        """Check if framework is valid using enum."""
        return fwk in [f.value for f in Framework]
    
    def get_framework_description(self) -> str:
        """Get description of current framework."""
        try:
            return Framework(self._framework).description
        except ValueError:
            return "Unknown framework"
    
    @classmethod
    def get_all_frameworks(cls) -> tuple[Framework, ...]:
        """Get all available frameworks from singleton."""
        return cls._config.frameworks
```

## Advanced Patterns

### Caching Enum Values for Performance

If you're validating frequently, cache the enum values:

```python
from pydasa.core.setup import Framework

class HighPerformanceClass:
    # Cache at class level - computed once
    _VALID_FRAMEWORKS = {f.value for f in Framework}
    _FRAMEWORK_VALUES = [f.value for f in Framework]
    
    @framework.setter
    @validate_choices(_FRAMEWORK_VALUES)
    def framework(self, val: str) -> None:
        self._framework = val
    
    def is_valid(self, fwk: str) -> bool:
        # O(1) lookup from cached set
        return fwk in self._VALID_FRAMEWORKS
```

### Using Singleton Globally

```python
from pydasa.core.setup import PyDASAConfig

# Get singleton instance once at module level
_CONFIG = PyDASAConfig.get_instance()

def validate_configuration(fwk: str, var_card: str) -> bool:
    """Validate framework and variable cardinality."""
    valid_fwks = {f.value for f in _CONFIG.frameworks}
    valid_cards = {c.value for c in _CONFIG.parameter_cardinality}
    
    return fwk in valid_fwks and var_card in valid_cards
```

## Backward Compatibility

The deprecated dictionary variables (`FDU_FWK_DT`, etc.) are still available for backward compatibility but will be removed in future releases. Update your code now to avoid breaking changes.

## Summary

- Import `PyDASAConfig` and specific `Enum` classes instead of dictionary variables
- Use `.value` to get string representation of enums
- Use `.description` to get human-readable descriptions
- Use singleton pattern via `PyDASAConfig.get_instance()` for configuration access
- Replace `@validate_choices(FDU_FWK_DT)` with `@validate_choices([f.value for f in Framework])`
- Replace default values like `"PHYSICAL"` with `Framework.PHYSICAL.value`
