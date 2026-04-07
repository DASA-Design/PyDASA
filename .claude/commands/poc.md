Scaffold or manage a proof of concept in lab/.

## Usage

- `/poc <name>` — Create a new PoC folder with a starter file
- `/poc list` — List existing PoCs and their status
- `/poc promote <name>` — Plan how to move a PoC into src/pydasa/ (does NOT auto-edit, proposes a plan)
- `/poc clean <name>` — Archive a dead-end PoC (logs it in devlog, optionally deletes)

## Creating a new PoC

1. Create `lab/<name>/` directory
2. Create `lab/<name>/<name>.py` with a header comment describing the hypothesis
3. Ask the user what they want to test
4. Write short, direct code — no abstractions, no packaging boilerplate, no docstrings
5. Add a devlog entry in `notes/devlog.md` with stage: `proof-of-concept`

## Rules

- Max 2 .py files + 1 supporting file per PoC
- Keep total code under ~500 lines
- No imports from src/pydasa/ unless testing integration with existing modules
- This is throwaway code — optimize for speed of exploration, not quality

## Promoting a PoC

When asked to promote:
1. Read the PoC code and identify what worked
2. Propose which module(s) in src/pydasa/ it should become or extend
3. List what needs to change (naming conventions, type hints, tests, validation)
4. Estimate the scope and update the devlog with stage: `planning`
5. Wait for approval before writing any production code
