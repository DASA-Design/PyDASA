Check or advance the stage of current work items.

## Usage

- `/stage` — Show all tracked work and their current stages
- `/stage <name> <new-stage>` — Update the stage of a work item

## Stages

```
proof-of-concept -> planning -> dev -> test -> analysis -> refinement -> merge
```

## What it does

1. Read `notes/devlog.md` and scan for entries with **Stage:** tags
2. Group by topic and show the latest stage for each
3. If advancing a stage, add a new devlog entry documenting the transition
4. When moving to `planning`: prompt for scope, affected modules, and approach
5. When moving to `dev`: verify there's a devlog planning entry, then proceed
6. When moving to `test`: run the test suite and report results
7. When moving to `merge`: run `/release-check` workflow

## Rules

- Stages only move forward (no skipping back without a new devlog entry explaining why)
- `merge` stage requires all tests passing and a devlog trail from PoC through refinement
- Not every item needs all stages — small fixes can start at `dev`
