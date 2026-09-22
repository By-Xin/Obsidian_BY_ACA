# E2E-Cardinality-Portfolio Slides

Beamer slides for:

> Hassan T. Anis and Roy H. Kwon, "End-to-end, decision-based, cardinality-constrained portfolio optimization," European Journal of Operational Research, 320(3), 739--753, 2025.

## Compile

Run from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error E2E-Cardinality-Portfolio-Slides.tex
```

Or run the full sequence manually:

```bash
pdflatex E2E-Cardinality-Portfolio-Slides.tex
biber E2E-Cardinality-Portfolio-Slides
pdflatex E2E-Cardinality-Portfolio-Slides.tex
pdflatex E2E-Cardinality-Portfolio-Slides.tex
```

The preamble still supports XeLaTeX/LuaLaTeX for system-font builds, but pdfLaTeX is the default because this deck has no Unicode body text that requires XeLaTeX. Do not rely on a single `pdflatex` pass after cleaning build files; unresolved biblatex citations will temporarily render as cite keys such as `[AgrawalAmosEtAl2019Layers]`.
