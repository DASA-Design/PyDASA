import numpy as np
from SALib.sample.fast_sampler import sample
from SALib.analyze.fast import analyze
import re


def generate_function_from_latex(latex_expr: str):
    """
    Generate a custom function from a LaTeX expression.

    Args:
        latex_expr (str): The LaTeX expression to convert.

    Returns:
        function: A Python function that evaluates the expression.
    """
    # Replace LaTeX-specific syntax with Python-compatible syntax
    python_expr = latex_expr.replace("^", "**")

    # Extract variable names from the LaTeX expression
    variables = sorted(set(re.findall(r"[a-zA-Z_]\w*", python_expr)))

    # Define the function dynamically using eval
    func_code = f"lambda {', '.join(variables)}: {python_expr}"
    custom_function = eval(func_code)

    return custom_function, variables


# Example LaTeX expression
latex_expression = "x^2 * y + y * z^3"

# Generate the custom function
custom_function, variables = generate_function_from_latex(latex_expression)

# Define the problem
problem = {
    "num_vars": len(variables),  # Number of variables
    "names": variables,  # Names of the variables
    "bounds": [[0, 1]] * len(variables),  # Bounds for each variable
}

# Generate samples using the FAST method
num_samples = 1000
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
