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

`app.js` drives three interactive figures. All of them progressively enhance the
static markup, so the page still reads with scripting disabled.

- **Guided self-attention explorer** — the stage chips highlight one of the feature
  maps in `assets/`, which are cropped straight out of the paper's GSA figure by
  `extract_gsa_assets.py` in the manuscript workspace.
- **GMP inference explorer** — an SVG rebuild of the paper's action-mode diagram.
  The component layout and mixture weights in `COMPONENTS` are schematic, matching
  the published diagram; they are not measured per-component values.
- **Semantic anchor bank** — shows what a mood word actually contributes to the
  prompt. `ANCHORS` is transcribed verbatim from `Real_robot/semantic_anchors.json`,
  and the displayed prompt follows `create_music_prompt()` in
  `Real_robot/music_creation.py` line for line, with the two anchor-written lines
  highlighted. `tempo_range` is shown as part of the record but is deliberately not
  in the prompt, because the prompt builder does not inject it. The seed line is the
  paper's example motif C4 E4 G4 as MIDI numbers.
  `verify_anchors.mjs` in the manuscript workspace drives the widget in headless
  Edge and diffs every displayed value against the JSON file.
