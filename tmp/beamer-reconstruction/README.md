# Beamer slide reconstruction

Current reconstruction and formatting requirements are recorded in
[`RECONSTRUCTION_RULES.md`](RECONSTRUCTION_RULES.md).

Place source photographs in `assets/`, ideally named in slide order:

- `slide-01.jpg`
- `slide-02.jpg`
- `slide-03.jpg`

Build with:

```bash
latexmk -xelatex main.tex
```

The deck is configured for 16:9 output and English text.
