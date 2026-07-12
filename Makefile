# WiFi CSI HAR — repeatable entry points. Repo rule: no hand-made plots;
# every report figure regenerates via `make figures`.

PY := .venv/bin/python

.PHONY: install test figures report clean-report

install:
	.venv/bin/pip install -e ".[dev,ml,demo]"

test:
	$(PY) -m pytest

figures:
	$(PY) -m csihar.figures

# Requires a TeX distribution (latexmk). Report compiles with placeholder
# boxes for figures that haven't been generated yet.
report:
	cd report && latexmk -pdf -interaction=nonstopmode main.tex

clean-report:
	cd report && latexmk -C main.tex
