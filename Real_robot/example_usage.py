"""
Example usage of the Co-policy music creation system
"""

import numpy as np
from music_creation import VLMMusicCreator

def demo_music_creation():
    """Demonstrate music creation without full system"""
    
    print("=== Co-policy Music Creation Demo ===")
    
    # Initialize the music creator
    creator = VLMMusicCreator(dry_run_robot=True)
    
    # Example 1: Create energetic music
    print("\n1. Creating energetic music...")
    result1 = creator.process_music_creation("Create an energetic music piece")
    print(f"Result: {result1}")
    
    # Example 2: Create peaceful melody
    print("\n2. Creating peaceful melody...")
    result2 = creator.process_music_creation("Play a peaceful melody")
    print(f"Result: {result2}")
    
    # Example 3: Create dramatic music
    print("\n3. Creating dramatic music...")
    result3 = creator.process_music_creation("Generate dramatic music")
    print(f"Result: {result3}")

def demo_note_extraction():
    """Demonstrate note extraction from audio"""
    from music_creation import MusicNoteExtractor
    
    print("\n=== Note Extraction Demo ===")
    
    extractor = MusicNoteExtractor()
    
    # Generate dummy audio data (sine wave)
    duration = 2.0  # seconds
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a simple melody (C-D-E-F-G)
    frequencies = [261.63, 293.66, 329.63, 349.23, 392.00]  # C4-G4
    audio_data = np.zeros_like(t)
    
    for i, freq in enumerate(frequencies):
        start_time = i * 0.4
        end_time = (i + 1) * 0.4
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        if end_idx <= len(t):
            audio_data[start_idx:end_idx] += np.sin(2 * np.pi * freq * t[start_idx:end_idx])
    
    # Extract notes
    notes = extractor.extract_notes(audio_data)
    print(f"Extracted {len(notes)} notes:")
    for i, (midi, duration, onset) in enumerate(notes):
        print(f"  Note {i+1}: MIDI {midi}, Duration {duration:.2f}s, Onset {onset:.2f}s")

def demo_gmp_prediction():
    """Demonstrate single-pass GMP action prediction"""
    from music_creation import GaussianMixtureVisuomotorPolicy
    
    print("\n=== GMP Policy Demo ===")
    
    gmp = GaussianMixtureVisuomotorPolicy()
    
    # Create dummy image (384x384x3)
    dummy_image = np.random.rand(384, 384, 3)
    
    # Test prediction for middle C (MIDI 60)
    try:
        joint_angles = gmp.predict_action(dummy_image, 60)
        print(f"Predicted executable action for MIDI 60 (C4): {joint_angles}")
    except Exception as e:
        print(f"GMP prediction error (expected without trained models): {e}")

if __name__ == "__main__":
    print("Co-policy Framework Example Usage")
    print("=" * 50)
    
    try:
        # Demo note extraction
        demo_note_extraction()
        
        # Demo GMP prediction
        demo_gmp_prediction()
        
        # Demo music creation (requires API keys and models)
        print("\nNote: Music creation demo requires:")
        print("- Qwen API key in configs.yaml")
        print("- Trained GMP or legacy GSA-compatible weights in models/ directory")
        print("- Robot connection, unless dry-run mode is used")
        
        # Uncomment to run full demo:
        # demo_music_creation()
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("Make sure all dependencies are installed and models are available.")