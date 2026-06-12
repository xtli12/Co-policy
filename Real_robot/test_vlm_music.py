"""
Test VLM music creation with proper Qwen API calls
"""

from handler_api import QwenVLConversation
import json

def test_qwen_music_generation():
    """Test Qwen VLM for music generation"""
    
    # Initialize conversation
    conversation = QwenVLConversation()
    
    # Create music generation prompt
    prompt = """Context: You are now a music creator. Based on the historical music knowledge you have learned and the user-provided requirements, along with the current notes, create music that aligns with the user's semantic input. Use "|" to separate the beats and output an array based on the notes.

Style Context: energetic rhythm - vivace, syncopated
Current note sequences: [Note60 | Note62 | Note64]
User command: Create an energetic music piece

Output JSON with fields: intent, style, tempo, human_seed, robot_role, available_notes, and robot_notes."""
    
    print("Sending prompt to Qwen VLM...")
    print(f"Prompt: {prompt}")
    
    try:
        # Call Qwen API
        response_stream = conversation.interact(
            user_input=prompt,
            user_image=None,
            use_history=False
        )
        
        # Collect response
        full_response = ""
        print("\nVLM Response Stream:")
        for sentence in response_stream:
            if sentence == "END":
                break
            print(f"  {sentence}")
            full_response += sentence + " "
        
        print(f"\nFull Response: {full_response}")
        
        # Extract MIDI notes from JSON or fallback text
        import re
        numbers = re.findall(r'\d+', full_response)
        midi_notes = [int(n) for n in numbers if 20 <= int(n) <= 108]
        
        print(f"Extracted MIDI notes: {midi_notes}")
        
        return midi_notes
        
    except Exception as e:
        print(f"Error calling Qwen API: {e}")
        return []

def test_music_creation_pipeline():
    """Test complete music creation pipeline"""
    
    from music_creation import VLMMusicCreator
    
    print("=== Testing Complete Music Creation Pipeline ===")
    
    try:
        # Initialize creator (without robot for testing)
        creator = VLMMusicCreator()
        
        # Test music creation
        result = creator.process_music_creation("Create an energetic music piece")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Pipeline test error: {e}")

if __name__ == "__main__":
    print("Testing VLM Music Creation")
    print("=" * 40)
    
    # Test 1: Direct Qwen API call
    print("\n1. Testing direct Qwen API call...")
    midi_notes = test_qwen_music_generation()
    
    # Test 2: Complete pipeline (comment out if no robot)
    print("\n2. Testing complete pipeline...")
    # test_music_creation_pipeline()
    
    print("\nTest completed!")