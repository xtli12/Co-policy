import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from handler_api import QwenVLConversation
from handler_speech import SpeechHandler
from GSA import SwinTransformer

try:
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
except Exception:  # pragma: no cover - robot SDK is optional for offline tests.
    RoboticArm = None
    rm_thread_mode_e = None


class MusicNoteExtractor:
    """Extract symbolic note events from acoustic streams.

    The paper treats audio as high-level note events. This extractor performs
    onset and pitch estimation with simple filtering, but it is not a full
    source-separation system for servo noise or chime reverberation.
    """

    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def extract_notes(self, audio_data: np.ndarray) -> List[Tuple[int, float, float]]:
        """Return a list of (midi_pitch, duration, onset_time) note events."""
        if audio_data is None or len(audio_data) == 0:
            return []

        audio_data = np.asarray(audio_data, dtype=np.float32)
        onset_frames = librosa.onset.onset_detect(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            backtrack=True,
        )
        onset_times = librosa.frames_to_time(
            onset_frames,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )

        pitches, magnitudes = librosa.piptrack(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )

        notes: List[Tuple[int, float, float]] = []
        for idx, onset_frame in enumerate(onset_frames):
            if onset_frame >= pitches.shape[1]:
                continue
            pitch_candidates = pitches[:, onset_frame]
            magnitude_candidates = magnitudes[:, onset_frame]
            if magnitude_candidates.size == 0 or magnitude_candidates.max() <= 0:
                continue

            max_idx = int(np.argmax(magnitude_candidates))
            freq = float(pitch_candidates[max_idx])
            if freq <= 0:
                continue

            midi_note = int(round(librosa.hz_to_midi(freq)))
            if not 20 <= midi_note <= 108:
                continue

            if idx < len(onset_times) - 1:
                duration = float(onset_times[idx + 1] - onset_times[idx])
            else:
                duration = 0.5
            notes.append((midi_note, duration, float(onset_times[idx])))

        return notes


class SemanticAnchorDB:
    """Semantic anchor bank used before VLM inference.

    Anchors are guardrails rather than a closed lookup table: they expose style,
    musical constraints, and robot playability tags to the VLM prompt.
    """

    def __init__(self, db_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = db_path or os.path.join(base_dir, "semantic_anchors.json")
        self.anchors = self._load_anchors()

    def _default_anchors(self) -> Dict[str, Dict]:
        return {
            "joyful": {
                "style_descriptor": "joyful melody",
                "technical_annotation": "allegro moderato, 4/4 time",
                "tempo_range": [120, 140],
                "robot_playability_tags": ["short motif", "clear onset", "reachable bells"],
            },
            "energetic": {
                "style_descriptor": "energetic rhythm",
                "technical_annotation": "vivace, syncopated rhythm",
                "tempo_range": [140, 180],
                "robot_playability_tags": ["limited leaps", "strong onsets", "stable tempo"],
            },
            "peaceful": {
                "style_descriptor": "peaceful harmony",
                "technical_annotation": "andante, legato phrasing",
                "tempo_range": [60, 80],
                "robot_playability_tags": ["slow tempo", "gentle strike", "small interval"],
            },
            "dramatic": {
                "style_descriptor": "dramatic expression",
                "technical_annotation": "forte, staccato expression",
                "tempo_range": [100, 120],
                "robot_playability_tags": ["accented strike", "clear rhythm", "safe reach"],
            },
        }

    def _load_anchors(self) -> Dict[str, Dict]:
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return self._default_anchors()

    def get_anchor(self, style_key: str) -> Dict:
        return self.anchors.get(style_key, self.anchors["joyful"])


class ConstrainedMusicPlanner:
    """Apply lightweight motif, novelty, and playability constraints."""

    def __init__(self, available_notes: Optional[List[int]] = None):
        self.available_notes = available_notes or [60, 62, 64, 65, 67, 69, 71]

    def constrain(self, generated_notes: List[int], human_seed: List[Tuple[int, float, float]]) -> List[int]:
        playable = [self._nearest_available(n) for n in generated_notes]
        if not playable and human_seed:
            playable = [self._nearest_available(n[0]) for n in human_seed]
        return playable[:16]

    def _nearest_available(self, note: int) -> int:
        return min(self.available_notes, key=lambda candidate: abs(candidate - note))


class GaussianMixtureVisuomotorPolicy:
    """Single-pass conditional mixture-density visuomotor policy.

    The expected trained GMP head outputs [logits, means, log_scales] for K
    latent action modes. For backward compatibility with earlier checkpoints,
    6D deterministic GSA weights are accepted as a legacy fallback.
    """

    def __init__(self, model_dir: str = "models/", num_modes: int = 6, action_dim: int = 6):
        self.model_dir = model_dir
        self.num_modes = num_modes
        self.action_dim = action_dim
        self.mdn_output_dim = num_modes * (1 + 2 * action_dim)
        self.models: Dict[object, Tuple[torch.nn.Module, str]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.note_models = {
            60: "C4_model.pth",
            62: "D4_model.pth",
            64: "E4_model.pth",
            65: "F4_model.pth",
            67: "G4_model.pth",
            69: "A4_model.pth",
            71: "B4_model.pth",
        }

    def _build_backbone(self, num_classes: int) -> torch.nn.Module:
        return SwinTransformer(
            hidden_dim=96,
            layers=(2, 2, 6, 2),
            heads=(3, 6, 12, 24),
            num_classes=num_classes,
        )

    def load_model(self, note: int) -> Tuple[torch.nn.Module, str]:
        if note in self.models:
            return self.models[note]

        model_file = self.note_models.get(note, "default_model.pth")
        model_path = os.path.join(self.model_dir, model_file)
        if not os.path.exists(model_path):
            model = self._build_backbone(self.action_dim).to(self.device).eval()
            self.models[note] = (model, "legacy_deterministic_untrained")
            return self.models[note]

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        for output_dim, kind in ((self.mdn_output_dim, "mixture_density"), (self.action_dim, "legacy_deterministic")):
            model = self._build_backbone(output_dim)
            try:
                model.load_state_dict(state_dict, strict=True)
                model.to(self.device).eval()
                self.models[note] = (model, kind)
                return self.models[note]
            except RuntimeError:
                continue

        raise RuntimeError(f"Checkpoint {model_path} does not match GMP or legacy GSA output dimensions")

    def predict_action(self, image: np.ndarray, note: int, selection: str = "expectation") -> List[float]:
        """Predict one executable six-DoF action in a single forward pass."""
        model, kind = self.load_model(note)
        image_tensor = self._preprocess_image(image)

        with torch.no_grad():
            output = model(image_tensor).view(-1)

        if kind == "mixture_density":
            return self._decode_mixture(output, selection=selection)
        return output[: self.action_dim].detach().cpu().numpy().astype(float).tolist()

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        if image.shape != (384, 384, 3):
            import cv2
            image = cv2.resize(image, (384, 384))
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        return image_tensor.to(self.device)

    def _decode_mixture(self, output: torch.Tensor, selection: str = "expectation") -> List[float]:
        k = self.num_modes
        d = self.action_dim
        logits = output[:k]
        means = output[k : k + k * d].view(k, d)
        # log_scales are retained for likelihood training/calibration; online
        # control only needs a mode or the mixture expectation.
        _log_scales = output[k + k * d : k + 2 * k * d].view(k, d)
        weights = F.softmax(logits, dim=0)
        if selection == "mode":
            action = means[int(torch.argmax(weights))]
        else:
            action = torch.sum(weights[:, None] * means, dim=0)
        return action.detach().cpu().numpy().astype(float).tolist()


# Backward-compatible alias for older scripts.
GSAMusicModel = GaussianMixtureVisuomotorPolicy


class RoboticMusicPerformer:
    """Robotic arm controller for chime performance."""

    def __init__(self, robot_ip: str = "192.168.1.19", robot_port: int = 8080, dry_run: bool = False):
        self.dry_run = dry_run or RoboticArm is None
        if self.dry_run:
            self.arm = None
            self.handle = None
            print("Robot SDK not available; running in dry-run mode.")
            return
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(robot_ip, robot_port)
        self.arm.rm_set_arm_max_line_speed(100)
        self.arm.rm_set_arm_max_line_acc(100)
        print(f"Robot connected with handle ID: {self.handle.id}")

    def execute_note(self, joint_angles: List[float]):
        """Execute a single note by moving to position and striking."""
        if self.dry_run:
            print(f"[dry-run] execute joint command: {joint_angles}")
            return
        self.arm.rm_movej(joint_angles, 100, 50, 1, 0)
        strike_angles = joint_angles.copy()
        strike_angles[-1] -= 7
        self.arm.rm_movej(strike_angles, 100, 50, 1, 0)
        self.arm.rm_movej(joint_angles, 100, 1, 0, 0)


class VLMMusicCreator:
    """Co-policy pipeline: semantic anchors, constrained planning, and GMP execution."""

    def __init__(self, dry_run_robot: bool = False):
        self.conversation = QwenVLConversation()
        self.speech_handler = SpeechHandler()
        self.note_extractor = MusicNoteExtractor()
        self.anchor_db = SemanticAnchorDB()
        self.planner = ConstrainedMusicPlanner()
        self.gmp_policy = GaussianMixtureVisuomotorPolicy()
        self.robot = RoboticMusicPerformer(dry_run=dry_run_robot)

    def create_music_prompt(self, user_command: str, notes: List[Tuple[int, float, float]], style_anchor: Dict) -> str:
        """Create a structured prompt for co-creation planning."""
        notes_str = " | ".join([f"Note{n[0]}" for n in notes]) or "None"
        playability = ", ".join(style_anchor.get("robot_playability_tags", []))
        return f"""You are a musical co-creator for a physical chime-playing robot.
Generate a complementary response rather than copying the human seed.
Return JSON with fields: intent, style, tempo, human_seed, robot_role, available_notes, and robot_notes.

Style anchor: {style_anchor['style_descriptor']} - {style_anchor['technical_annotation']}
Robot playability constraints: {playability}
Current human seed notes: [{notes_str}]
User command: {user_command}
"""

    def extract_music_from_response(self, response: str) -> List[int]:
        """Extract robot MIDI notes from a VLM JSON response or fallback text."""
        try:
            start = response.index("{")
            end = response.rindex("}") + 1
            payload = json.loads(response[start:end])
            notes = payload.get("robot_notes", [])
            return [int(n) for n in notes if 20 <= int(n) <= 108]
        except Exception:
            numbers = re.findall(r"\d+", response)
            return [int(n) for n in numbers if 20 <= int(n) <= 108]

    def perform_music(self, midi_notes: List[int], image_path: Optional[str] = None):
        """Perform a note sequence with single-pass GMP action inference."""
        if image_path is None:
            image = np.zeros((384, 384, 3), dtype=np.float32)
        else:
            image = np.load(image_path) if image_path.endswith(".npy") else np.zeros((384, 384, 3), dtype=np.float32)

        print(f"Performing {len(midi_notes)} notes...")
        for idx, note in enumerate(midi_notes):
            print(f"Playing note {idx + 1}/{len(midi_notes)}: MIDI {note}")
            joint_angles = self.gmp_policy.predict_action(image, note)
            self.robot.execute_note(joint_angles)
            time.sleep(0.2)

    def process_music_creation(self, user_input: str, audio_data: Optional[np.ndarray] = None) -> str:
        """Run semantic grounding, constrained planning, and robotic execution."""
        try:
            current_notes = self.note_extractor.extract_notes(audio_data) if audio_data is not None else []
            style_key = self._infer_style(user_input)
            style_anchor = self.anchor_db.get_anchor(style_key)
            prompt = self.create_music_prompt(user_input, current_notes, style_anchor)

            response_stream = self.conversation.interact(user_input=prompt, user_image=None, use_history=False)
            full_response = ""
            for sentence in response_stream:
                if sentence == "END":
                    break
                full_response += sentence + " "

            print(f"VLM Response: {full_response}")
            generated_notes = self.extract_music_from_response(full_response)
            midi_notes = self.planner.constrain(generated_notes, current_notes)

            if not midi_notes:
                return "Could not generate valid playable notes from VLM response."

            print(f"Playable MIDI notes: {midi_notes}")
            self.perform_music(midi_notes)
            return f"Created and performed a constrained co-creative response with {len(midi_notes)} notes: {midi_notes}"
        except Exception as exc:
            print(f"Error in music creation: {exc}")
            return f"Music creation failed: {exc}"

    @staticmethod
    def _infer_style(user_input: str) -> str:
        text = user_input.lower()
        if "energetic" in text:
            return "energetic"
        if "peaceful" in text:
            return "peaceful"
        if "dramatic" in text:
            return "dramatic"
        return "joyful"


def main():
    creator = VLMMusicCreator(dry_run_robot=True)
    print("Co-policy music co-creation system ready.")
    print("Commands: 'create energetic music', 'play peaceful melody', etc.")
    while True:
        try:
            user_input = input("\nEnter music creation command (or 'quit' to exit): ")
            if user_input.lower() == "quit":
                break
            result = creator.process_music_creation(user_input)
            print(f"Result: {result}")
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
