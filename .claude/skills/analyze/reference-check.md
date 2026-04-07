# Reference Check Skill

Use this skill when mapping references between a PDF chapter and `db/references.bib`.

## Purpose

Each chapter PDF uses numbered references [1], [2], etc. The BibTeX file uses cite keys like `author_keyword_year`. This skill produces a 1-to-1 mapping between them and identifies gaps.

## Process

1. **Extract** all numbered references from the PDF chapter (author, year, title)
2. **Search** `db/references.bib` for each reference by author last name and year
3. **Map** each [N] to its BibTeX key — every reference must have exactly one match
4. **Flag missing** references not found in the bib file with `% TODO: add to Zotero`
5. **Flag orphans** — bib entries that exist but are not used in the chapter (optional, on request)

## Output Format

Produce two tables in the tgt notes under `## Reference Mapping`:

### Mapped References

| Ref # | BibTeX Key | Author | Year |
|---|---|---|---|
| [1] | `cite_key` | Author et al. | 2017 |

### Missing from BibTeX — needs Zotero import

| Ref # | Author | Year | Title | DOI |
|---|---|---|---|---|
| [10] | Braten et al. | 2021 | Title here | doi:xxx |

## Rules

- Every [N] must map to exactly one BibTeX key — no duplicates, no gaps
- Search by author last name first, then narrow by year and title keywords
- If multiple bib entries match the same reference, flag the ambiguity
- If a bib entry has a slightly different title or author spelling, still map it and note the discrepancy
- Never modify `db/references.bib` — it is read only and managed by Zotero
- The mapping goes in the chapter's tgt notes and in the chapter skill under `## Key references`

## Search Strategy

1. Search `@` entries in bib file by first author last name (case insensitive)
2. If not found, search by second author or title keywords
3. If still not found, search by DOI
4. If no match at all, mark as `% TODO: add to Zotero` with full bibliographic entry and DOI for easy import
