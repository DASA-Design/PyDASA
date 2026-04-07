# Notebook Editing Skill

Use this skill when editing Jupyter notebooks (.ipynb files).

## Rules

- Always confirm the exact cell number/index before making changes
- Never edit adjacent or similarly-named cells without confirmation
- If unsure which cell to modify, list all cells with their index and first line, then ask
- Make only the specific changes requested in the target cell
- Do not refactor, reorganize, or "improve" surrounding cells
- Do not add unrequested changes to other cells

## Before Editing

1. Read the notebook to understand cell structure
2. Identify the exact cell by index number and content
3. Confirm with the user if there is any ambiguity
4. Only then proceed with the edit

## Cell Identification

- Reference cells by their index number (e.g., "cell 14")
- When the user provides a selection, verify it matches the cell content
- Markdown cells in notebooks contain valuable context — read them before editing code cells
