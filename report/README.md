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

## Status (2026-08-28)

| Chapter | State |
|---|---|
| 1 Introduction | written |
| 2 Literature Review | written |
| 3 Methodology | written |
| 4 System Design | written |
| 5 Results | skeleton — needs real-data runs |
| 6 Conclusion | Limitations + Future Work written; Conclusion needs results |
| Abstract | last |

Chapters 1–4 are data-independent: they describe the system and the
protocol, both of which are frozen. Every measured number in them
(acquisition rate, null subcarriers, latency, channel survey) comes from a
real capture and is sourced in CLAUDE.md's hard-won facts. Two blanks remain
that only hardware can fill: the receiver-layout figure (§4.1) and the
system-architecture diagram (§3.1), both currently placeholder boxes.

Do not write the Results chapter from pilot data. One subject on one day
cannot produce an interpretable cross-session or cross-subject number, which
is the whole point of the evaluation protocol Chapter 3 commits to.

## Citations

`references.bib` seeds only verified entries; two have TODO fields to
complete from publisher pages. Add the proposal's refs [1]–[9] as cited.
