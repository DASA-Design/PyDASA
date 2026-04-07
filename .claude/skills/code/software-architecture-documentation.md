# Software Architecture Documentation Skill

This skill produces rigorous, structured architecture documentation suitable for technical reports, dissertation chapters, and system design documents. It follows the ADD (Attribute-Driven Design) method of Bass, Clements & Kazman as the default structure, with C4 and arc42 as alternatives.

---

## Step 0 — Establish Context Before Writing

Before producing any documentation, gather the following. Extract what you can from the conversation or uploaded files, then ask for what is missing.

1. **System identity** — name, version, language, licence, repository
2. **Primary purpose** — what problem the system solves, for whom
3. **Documentation target** — dissertation chapter, technical report, internal wiki, public docs?
4. **Audience** — academic examiners, developers, architects, non-technical stakeholders?
5. **Framework preference** — ADD (default), C4, arc42, or none?
6. **Edition of Bass et al.** — 3rd edition (7 QAs) or 4th edition (10 QAs)? Default to 3rd unless specified.
7. **Available artefacts** — code, existing docs, diagrams, README, prior drafts?

Do not start writing until items 1-4 are confirmed.

---

## Default Structure — ADD (Attribute-Driven Design)

The standard chapter structure follows ADD. Adapt section numbers to the surrounding document.

```
X.1  Introduction
X.2  Key Functional Requirements
X.3  Critical Quality Attributes (Utility Tree)
X.4  Design Principles and Concerns
X.5  Software Architecture (Views)
     X.5.1  Context Diagram
     X.5.2  Information Flow Diagram
     X.5.3  Process Diagram(s)
     X.5.4  Module / Component / Class Diagram
     X.5.5  Configuration / Schema Diagram
X.6  Implementation Details
X.7  Installation and Usage Tutorial
X.8  Limitations and Future Development
```

---

## Functional Requirements

Derive FRs from the system's user stories or stated purpose. Format as numbered requirements with a short label:

```
FR-1: [Label]. The system shall [capability]. [One sentence of scope or constraint if needed.]
FR-2: [Label]. The system shall [capability].
```

Rules:
- Each FR maps to one capability. Split compound requirements.
- FRs are testable — avoid "the system should be easy to use" (that belongs in quality attributes).
- Use "shall" for mandatory requirements.
- Keep each FR to two sentences maximum.

---

## Utility Tree

Produce a utility tree following Bass, Clements & Kazman. Leaf nodes are ASR scenarios rated `[Business Importance / Technical Risk]` where H = High, M = Medium, L = Low.

### Bass 3rd Edition Quality Attributes (default)

1. Availability
2. Interoperability
3. Maintainability
4. Performance
5. Security
6. Testability
7. Usability

### Bass 4th Edition Quality Attributes

All 3rd edition attributes plus: Deployability, Energy Efficiency, Integrability.

### Utility Tree Format

```
Utility
|
+-- ATTRIBUTE  [* if primary]
|   +-- Refinement Area
|   |   +-- [H/H] Scenario: stimulus -> measurable response
|   |   +-- [M/L] Scenario: stimulus -> measurable response
|   +-- Refinement Area
|       +-- [H/M] Scenario: stimulus -> measurable response
|
+-- ATTRIBUTE
    +-- ...
```

### Rules for Good ASR Scenarios

Each scenario must include:
- A **stimulus** (who does what, or what event occurs)
- A **measurable response** (what the system does, with a quantitative measure where possible)

Good: `run_analysis() with a singular matrix raises a descriptive exception within 1 second`
Bad: `the system handles errors gracefully`

Keep scenarios to two lines maximum. Do not repeat the same scenario under multiple attributes.

Mark 2-4 attributes as primary (*) — those that most directly shaped architectural decisions.

---

## Architectural Views

Produce views appropriate to the system. Each is produced as a descriptive placeholder with a clear caption that specifies exactly what the diagram must show.

### Context Diagram (C4 Level 1 or equivalent)

Describe: the system at the centre, external actors (users, other systems), and the nature of each interaction. State explicitly which interactions are runtime and which are development-time only.

### Information Flow Diagram

Describe: how data moves through the system's internal layers during a representative operation. Use a numbered sequence or data flow notation. Distinguish between control flow and data flow where relevant.

### Process Diagram

For each significant pipeline or workflow, produce one process diagram. Use UML Activity or BPMN 2.0 notation descriptions. Specify:
- Start and end events
- Decision gateways and their conditions
- Swimlanes if multiple actors are involved
- Exception/error paths

### Module / Component / Class Diagram

Describe: the internal structure of the system. Include:
- Module or package hierarchy with responsibility annotations
- Dependency direction (always specify — "arrows point from dependent to dependency")
- Test coverage status if relevant
- Key classes with their principal attributes and methods in UML notation

### Configuration / Schema Diagram

Use when the system supports multiple configurations, schemas, plugins, or framework variants. Describe each variant and show how the same core engine/component processes all of them.

---

## Design Principles

Document each design principle as a named principle with its trade-off explicitly stated:

```markdown
**Principle Name.** What the principle requires and why it was chosen.
The trade-off is [what is sacrificed or complicated by this choice].
```

Aim for 4-6 principles. Each should be traceable to a primary quality attribute.

---

## Voice and Style

- Use first person plural ("we") for design decisions made by the authors.
- Use third person or named attribution for external tools, frameworks, and prior work ("Bass et al. define...", "NumPy provides...").
- Avoid em dashes. Use commas, full stops, colons, or semicolons instead.
- Use proper connective transitions between paragraphs.
- Write in declarative sentences. Avoid hedging language in requirement and design statements.

---

## Limitations and Future Work

Document limitations honestly, grouped by type:

1. **Functional gaps** — features not yet implemented (stub modules, pending development)
2. **Validation gaps** — aspects not yet empirically validated
3. **Stability** — API stability, versioning policy, reproducibility instructions

Each limitation entry: state the gap, explain why it exists (design decision vs. incomplete work), and state how affected use cases are currently handled.

---

## Utility Tree Scenario Patterns

Use these patterns to construct well-formed ASR scenarios. Each pattern shows the stimulus and measurable response structure.

**Pattern:** `[Actor] [action] -> system [response] within [measure]`

### Availability
- `[Component] is called with [invalid input]; system raises a descriptive exception within [N] seconds identifying the [offending element]`
- `[Validation mechanism] rejects [ill-formed input] at [instantiation] time, before [harmful state propagates]`

### Interoperability
- `A [domain object] exported via [method()] is used to construct a [external data structure] in [N lines] with no [custom adapter]`
- `[Output format] rendered by [module] is consumed by [external tool] without [manual post-processing]`

### Maintainability
- `A developer adds [new variant] by supplying [a config file]; no changes to [module A], [module B], or [module C] are required`
- `A new [variant] passes the existing test suite with only [new fixture data] added`

### Performance
- `A [N-variable] [analysis type] completes in under [T] seconds on a [standard laptop] with no [external compute]`

### Security
- `A [malformed input] passed as [parameter] raises a [named exception]; no [data corruption] occurs`

### Testability
- `[Module] is independently importable and fully testable without instantiating [a full component] or any [external dependency]`
- `[Known scenario] produces exactly [expected output]; this is used as a [regression oracle] for the full pipeline`

### Usability
- `A [user type] can [accomplish goal] by following the [N-step quickstart] without reading the [full API reference]`

### Rating Guide

| Rating | Business Importance | Technical Risk |
|--------|---------------------|----------------|
| H | Failure would critically impair research validity or user trust | No known solution; significant effort required |
| M | Failure would degrade experience but not invalidate results | Solution exists but requires non-trivial work |
| L | Nice to have; failure has minimal impact | Straightforward to implement |

At most 30% of scenarios should be [H/H].

---

## Section Writing Guidance

### Introduction (X.1)
4-5 paragraphs: prior context, gap identification, system introduction, chapter roadmap.

### Functional Requirements (X.2)
Numbered FR-1 through FR-N. One capability per FR. Testable, "shall" language.

### Quality Attributes (X.3)
Utility tree with rated scenarios. 3+ scenarios per primary attribute. Closing paragraph on why these attributes were chosen.

### Design Principles (X.4)
4-6 named principles with explicit trade-offs. Each traceable to a primary QA.

### Architecture Views (X.5)
Open with diagram index table. Context, information flow, process, module/class, configuration views as appropriate.

### Implementation Details (X.6)
Dependencies table, dev environment, dev process. Trace commit convention to version increment.

### Tutorial (X.7)
Prerequisites, install, worked example with verifiable output, API workflow summary table.

### Limitations (X.8)
Functional gaps, validation gaps, stability. Close with version string and reproducibility instruction.
