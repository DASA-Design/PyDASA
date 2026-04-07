Add or review entries in the development log (notes/devlog.md).

## Usage

- `/devlog` — Show recent entries (last 3)
- `/devlog add` — Add a new entry for today's work (interactive)
- `/devlog <name>` — Show entries related to a specific topic/PoC
- `/devlog status` — List all open experiments (not yet accepted/abandoned)

## Adding an entry

1. Read `notes/devlog.md` to see the existing format and latest entries
2. Ask the user: what did we try, what happened, what's the decision?
3. Write the entry with this structure:

```markdown
## YYYY-MM-DD | <topic> - <short description>

**Stage:** proof-of-concept | planning | dev | test | analysis | refinement | accepted | abandoned

**Hypothesis:** What we expected or wanted to test

**What was done:**
- Bullet points of actions taken

**Result:**
- What worked, what didn't, measurements if any

**Decision:** accepted / abandoned / pending — <reasoning>
```

4. If a previous entry exists for the same topic, link them (e.g., "continues from 2026-04-06")

## Rules

- Use today's actual date, never a placeholder
- Be specific about what was tried — future-you needs to understand without context
- Dead ends are valuable: document WHY it didn't work, not just that it didn't
- This is NOT a changelog — entries here may never become features or commits
