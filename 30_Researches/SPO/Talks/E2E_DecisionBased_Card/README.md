# E2E-Cardinality-Portfolio Slides

Beamer slides for:

> Hassan T. Anis and Roy H. Kwon, "End-to-end, decision-based, cardinality-constrained portfolio optimization," European Journal of Operational Research, 320(3), 739--753, 2025.

## Compile

Run from this directory:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error E2E-Cardinality-Portfolio-Slides.tex
```

Or run the full sequence manually:

```bash
xelatex E2E-Cardinality-Portfolio-Slides.tex
biber E2E-Cardinality-Portfolio-Slides
xelatex E2E-Cardinality-Portfolio-Slides.tex
xelatex E2E-Cardinality-Portfolio-Slides.tex
```

Do not rely on a single `xelatex` pass after cleaning build files; unresolved biblatex citations will temporarily render as cite keys such as `[AgrawalAmosEtAl2019Layers]`.
