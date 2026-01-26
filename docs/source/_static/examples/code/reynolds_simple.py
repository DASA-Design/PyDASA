"""
Simple Reynolds Number Analysis with PyDASA
===========================================

This example shows the minimal steps to:
1. Define physical variables with dimensions (using dicts)
2. Run dimensional analysis using Buckingham Pi theorem
3. Get the Reynolds number as a dimensionless coefficient

Reynolds Number: Re = (ρ·v·L)/μ
"""

# Import PyDASA
from pydasa.workflows.phenomena import AnalysisEngine

# Step 1: Define your physical variables as dictionaries
# PyDASA will automatically convert them to Variable objects
# NOTE: At least one variable must be an OUTPUT (cat='OUT')
variables = {
    # Density: ρ [M/L³] - INPUT
    "\\rho": {
        "_idx": 0,
        "_sym": "\\rho",
        "_fwk": "PHYSICAL",
        "_cat": "IN",
        "_name": "Density",
        "relevant": True,
        "_dims": "M*L^-3",
        "_units": "kg/m³",
        "_setpoint": 1000.0,  # water
        "_std_setpoint": 1000.0,  # water
    },

    # Velocity: v [L/T] - OUTPUT (what we're predicting/analyzing)
    "v": {
        "_idx": 1,
        "_sym": "v",
        "_fwk": "PHYSICAL",
        "_cat": "OUT",  # At least one variable must be OUTPUT
        "_name": "Velocity",
        "relevant": True,
        "_dims": "L*T^-1",
        "_units": "m/s",
        "_setpoint": 2.0,
        "_std_setpoint": 2.0,
    },

    # Length: L [L] - INPUT
    "L": {
        "_idx": 2,
        "_sym": "L",
        "_fwk": "PHYSICAL",
        "_cat": "IN",
        "_name": "Length",
        "relevant": True,
        "_dims": "L",
        "_units": "m",
        "_setpoint": 0.05,
        "_std_setpoint": 0.05,
    },

    # Viscosity: μ [M/(L·T)] - INPUT
    "\\mu": {
        "_idx": 3,
        "_sym": "\\mu",
        "_fwk": "PHYSICAL",
        "_cat": "IN",
        "_name": "Viscosity",
        "relevant": True,
        "_dims": "M*L^-1*T^-1",
        "_units": "Pa·s",
        "_setpoint": 0.0001,  # water
        "_std_setpoint": 0.0001,  # water
    }
}

# Step 2: Create the Analysis Engine and add variables
# AnalysisEngine automatically converts dict to Variable objects
engine = AnalysisEngine(
    _idx=0,
    _fwk="PHYSICAL",
    _name="Reynolds Number Analysis"
)
engine.variables = variables  # PyDASA converts dicts to Variable objects

# Step 3: Run the dimensional analysis
results = engine.run_analysis()

# Step 4: Display results
print("=" * 50)
print("DIMENSIONAL ANALYSIS RESULTS")
print("=" * 50)
print(f"Number of dimensionless groups: {len(engine.coefficients)}")
print()

for name, coeff in engine.coefficients.items():
    print(f"{name}: {coeff.pi_expr}")
    print(f"  Variables: {list(coeff.var_dims.keys())}")
    print(f"  Exponents: {list(coeff.var_dims.values())}")
    print()

# Step 5: Derive Reynolds Number using AnalysisEngine.derive_coefficient()
# Get the Pi_0 key (inverse Reynolds)
pi_0_key = list(engine.coefficients.keys())[0]

# Use derive_coefficient() to create Reynolds Number (inverse of Pi_0)
Re_coeff = engine.derive_coefficient(
    expr=f"1/{pi_0_key}",  # Reynolds = 1/Pi_0
    symbol="Re",
    name="Reynolds Number",
    description="Reynolds number: Re = (ρ·v·L)/μ"
)

# Calculate the numerical value using the Coefficient's calculate_setpoint() method
Re_value = Re_coeff.calculate_setpoint()

print("=" * 50)
print(f"Derived Coefficient: {Re_coeff.sym} = {Re_coeff.pi_expr}")
print(f"Reynolds Number (Re) = {Re_value:.2e}")
print("=" * 50)

# Interpret the result
if Re_value < 2300:
    print("Flow regime: LAMINAR")
elif Re_value < 4000:
    print("Flow regime: TRANSITIONAL")
else:
    print("Flow regime: TURBULENT")

print("=" * 50)
