# Co-policy Promotional Page

This directory contains a static promotional page for the Co-policy paper.

## Local Preview

From the repository root:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/docs/
```

## Video Asset

The page expects the paper demonstration video at the repository root:

```text
Co-policy_Video_Demonstration.mp4
```

Demo GIFs are loaded from the public GitHub repository.

## Interactive Figures

`app.js` drives two interactive figures. Both progressively enhance the static
markup, so the page still reads with scripting disabled.

- **Guided self-attention explorer** — the stage chips highlight one of the feature
  maps in `assets/`, which are cropped straight out of the paper's GSA figure by
  `extract_gsa_assets.py` in the manuscript workspace.
- **GMP inference explorer** — an SVG rebuild of the paper's action-mode diagram.
  The component layout and mixture weights in `COMPONENTS` are schematic, matching
  the published diagram; they are not measured per-component values.
