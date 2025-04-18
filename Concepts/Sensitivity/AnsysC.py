import numpy as np
from SALib.sample.fast_sampler import sample
from SALib.analyze.fast import analyze
from sympy.parsing.latex import parse_latex
from sympy import lambdify  # , symbols
# import re
# Beware pip install should be: pip install antlr4-python3-runtime==4.11


def generate_function_from_latex(latex_expr: str):
    """
    Generate a custom function from a LaTeX expression.

    Args:
        latex_expr (str): The LaTeX expression to convert.

    Returns:
        function: A Python function that evaluates the expression.
        variables: List of variables in the function.
    """
    # Parse the LaTeX expression into a sympy expression
    sympy_expr = parse_latex(latex_expr)

    # Extract variables from the sympy expression
    variables = sorted(sympy_expr.free_symbols, key=lambda s: s.name)

    # Generate a callable function using lambdify
    custom_function = lambdify(variables, sympy_expr, "numpy")

    return custom_function, [str(v) for v in variables]


# Define LaTeX expressions
latex_expressions = [
    r"\frac{u}{U}",
    r"\frac{y*P}{U^2.0}",
    r"\frac{v}{y*U}"
]

# Process each LaTeX expression to generate functions
functions = []
problems = []

for latex_expr in latex_expressions:
    custom_function, variables = generate_function_from_latex(latex_expr)

    # Define the problem for SALib
    problem = {
        "num_vars": len(variables),  # Number of variables
        "names": variables,  # Names of the variables
        "bounds": [[0.1, 10]] * len(variables),  # Bounds for each variable
    }

    functions.append((custom_function, variables))
    problems.append(problem)

# Perform sensitivity analysis for each function
num_samples = 1000

for i, (custom_function, variables) in enumerate(functions):
    print(f"\nAnalyzing Function {i + 1}: {latex_expressions[i]}")

    # Get the corresponding problem definition
    problem = problems[i]

    # Generate samples using the FAST method
    param_values = sample(problem, num_samples)

    # Reshape the samples to match the expected input format for the custom function
    param_values = param_values.reshape(-1, problem["num_vars"])

    # Evaluate the custom function for all samples
    Y = np.apply_along_axis(lambda row: custom_function(*row), 1, param_values)

    # Perform sensitivity analysis using FAST
    Si = analyze(problem, Y)

    # Print results
    print("Sensitivity Indices:")
    for name, S1 in zip(problem["names"], Si["S1"]):
        print(f"  {name}: {S1:.6f}")
