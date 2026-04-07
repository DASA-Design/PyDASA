# Debugging Skill

Use this skill when debugging Python code or Jupyter notebooks.

## Rules

- Read the actual code structure carefully before suggesting fixes
- Do not assume code structure — verify variable names, dictionary key formats, and data types by reading the relevant cells or files first
- Never suggest fixes based on assumptions about how the code "should" look

## Before Fixing

1. Read the relevant code cells or files completely
2. List all dictionary keys, variable names, and their exact formats
3. Identify the root cause before proposing a fix
4. Show the user what needs to change and wait for approval before editing

## Common Pitfalls to Avoid

- Dictionary key format mismatches (e.g., missing braces, wrong prefixes)
- Deep copy vs shallow copy issues in data mutation
- Python for-else indentation bugs
- Object dtype vs numeric dtype in pandas DataFrames
- Wrong cell being edited when fixing notebook bugs
