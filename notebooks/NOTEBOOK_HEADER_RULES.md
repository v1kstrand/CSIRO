# Skill: Notebook Header Structure

## Purpose
Define a consistent, lightweight header scheme for all notebooks in this repo so navigation is fast and sections are predictable.

## Scope
Applies to every notebook under `notebooks/` regardless of topic (EDA, training, inference, reporting).

## Inputs
- A notebook and its intended flow (setup, data loading, analysis, results).

## Outputs
- A notebook with standardized headers and section structure.

## Rules
1) Use `#` only for major chapters. Target 2–4 per notebook.
2) Use `##` for sub-sections only when a chapter has 3+ cells or clearly distinct steps.
3) Use `###` only for deeper splits when a `##` section has 3+ cells.
4) Helper/utility cells should not start new chapters; keep them under the current `#` or `##`.
5) Avoid redundant headers; if a single cell can stand alone, it can have no header.
6) Keep header titles short and action-oriented (e.g., “Setup & Imports”, “Fold Predictions”, “Diagnostics”).
7) Keep ordering linear: setup → data → modeling → evaluation/EDA → conclusions.
8) Keep headers stable across notebooks so they’re easy to scan and compare.

## Template
```
# <Main Section 1>
## <Subsection 1>   (optional)
## <Subsection 2>   (optional)

# <Main Section 2>
## <Subsection 1>   (optional)
### <Sub-sub>       (optional)

# <Main Section 3>
```

## Example (from pred_eda.ipynb)
```
# OOF EDA: Setup & Imports

# OOF EDA: Fold Predictions

# OOF EDA: Diagnostics
```

## Success Criteria
- Notebook outline is navigable with 2–4 `#` headers.
- Subsections only appear when they reduce scrolling (3+ related cells).
- Helper cells do not create new chapters.
