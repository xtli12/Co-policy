# Co-policy Real-Robot Music Co-Creation

This directory contains the real-robot implementation skeleton for the paper's Co-policy framework. The code follows the current paper narrative: semantic intent grounding with pre-inference anchors, constrained musical variation, and single-pass Gaussian-Mixture Visuomotor Policy (GMP) execution.

## System Components

1. **Semantic anchors** (`semantic_anchors.json`): style, tempo, and robot-playability guardrails used before VLM inference.
2. **VLM planner** (`music_creation.py`): prompts Qwen-VL to produce a structured co-creation plan instead of unconstrained prose.
3. **Constrained music planner**: maps generated notes to playable chime notes and keeps the response physically executable.
4. **GMP execution policy**: `GaussianMixtureVisuomotorPolicy` predicts executable six-DoF actions in one forward pass. A trained GMP head may output mixture weights, means, and scales over latent action modes; older 6D GSA checkpoints are supported as a deterministic compatibility fallback.
5. **Robot performer**: sends the selected action to the Realman arm controller. If the robot SDK is unavailable, the code can run in dry-run mode for offline testing.

## Installation

```bash
pip install -r requirements_music.txt
```

Expected local resources:

- `speech_recognition_v2/configs.yaml`: Qwen/DashScope API configuration.
- `models/`: trained GMP checkpoints, or legacy GSA-compatible 6D checkpoints.
- `Robotic_Arm/`: Realman robotic arm SDK or interface package.

## Quick Start

Run the deployed loop:

```bash
python co_policy_main.py
```

Run component demos without a full robot setup:

```bash
python example_usage.py
```

Run a direct VLM prompt test:

```bash
python test_vlm_music.py
```

## Current Inference Flow

1. The user provides speech and optionally live musical seed notes.
2. `MusicNoteExtractor` converts audio into high-level note events after onset/pitch filtering. It is not a full acoustic source-separation system for servo noise or chime reverberation.
3. `SemanticAnchorDB` retrieves style and playability anchors.
4. Qwen-VL receives a structured prompt and returns a JSON-like co-creation plan with robot notes.
5. `ConstrainedMusicPlanner` keeps the generated notes within reachable/playable chime notes.
6. `GaussianMixtureVisuomotorPolicy` maps the target note and visual observation to a six-DoF robot action in a single forward pass.
7. `RoboticMusicPerformer` executes the strike and returns to the nominal posture.

## Notes on Paper Alignment

- GSA is treated as the visual encoder/backbone for the action policy, not as a standalone initial-action regressor.
- GMP is a conditional mixture-density policy over full action vectors or short trajectory segments.
- Mixture components are latent action modes, not individual robot joints.
- Online execution uses one forward pass through the policy and does not perform mixture re-estimation or distance-based post-processing.
- Post-command action frequency measures the visuomotor execution loop, not full speech-to-action latency.

## Limitations

The current implementation uses high-level audio note events and limited contact feedback. Servo noise, chime reverberation, and mis-strikes are not fully handled by a low-level tactile/force feedback loop. These limitations match the discussion in the paper and should be addressed with acoustic source separation, contact sensing, and recovery policies in future work.
