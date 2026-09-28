# Paper figures

These scripts read real experiment artifacts from `results/`; they never embed,
estimate, smooth, or repair result values. Each figure is rendered from one
canvas to PDF, SVG, and 500 dpi PNG, then checked against its `figure.yaml`.
The scripts require an exact Arial installation and fail rather than silently
substituting another font.

From the project root:

```bash
python fig/generate_all.py
```

The generated files and `qa-report.json` stay beside each plotting script and
are ignored by Git because they can be reproduced from the result artifacts.
