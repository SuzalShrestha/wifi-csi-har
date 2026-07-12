# Report (Phase 6)

LaTeX skeleton for the EX 707 major-project report. **Verify the exact
department format** (title/certificate pages, margins, citation style)
against current TU/IOE guidelines before submission — this skeleton is
generic.

## Build

```bash
make report          # latexmk -pdf report/main.tex (needs a TeX distribution)
make clean-report
```

Compiles at any project stage: figures not yet generated render as labeled
placeholder boxes.

## Figures

All figures live in `docs/figures/` and regenerate from results CSVs:

```bash
make figures         # python -m csihar.figures
```

Never paste hand-made or screenshot plots (repo rule; see CLAUDE.md).
Confusion matrices are written per training run by `csihar.train` /
`csihar.experiments`.

## Writing order (from IMPLEMENTATION_PLAN.md)

Methodology and system-design chapters can be written NOW — the software is
done and doesn't change with data. Results/failure-analysis chapters wait for
real-data runs. Abstract last.

## Citations

`references.bib` seeds only verified entries; two have TODO fields to
complete from publisher pages. Add the proposal's refs [1]–[9] as cited.
