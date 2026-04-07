Generate a code review report for a module, package, or the full codebase.

Read the skill at `.claude/skills/code/code-review-report.md` and follow it exactly.

## Usage

- `/code-report <module.py>` — Analyze a single file
- `/code-report <package>` — Analyze a package directory (e.g., `dimensional`, `workflows`)
- `/code-report full` — Analyze the entire `src/pydasa/` codebase

## Steps

1. Verify the venv is active: run `echo $VIRTUAL_ENV` and confirm it points to this project. If not, activate with `source venv/Scripts/activate` first.
2. Read `.claude/skills/develop/coding-conventions.md` so you know what NOT to flag
2. Read all `.py` files in the target scope
3. Produce the seven-section report: Abstractions, Dependencies, Good Ideas, Risks, Bugs and Complications, Technical Debt, Summary
4. Save the report to `notes/reports/code_review_<scope>_<date>.md`
5. Add a devlog entry noting the report was generated

Every finding must cite file:lines. No vague claims. Do not suggest fixes — diagnosis only.
