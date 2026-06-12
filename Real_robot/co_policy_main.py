import time
import numpy as np
from handler_chat import ChatHandler
from music_creation import VLMMusicCreator

class CoPolicyFramework:
    """Main Co-policy framework integrating VLM planning and GMP execution."""
    
    def __init__(self):
        self.chat_handler = ChatHandler()
        self.music_creator = VLMMusicCreator()
        
        # Wake and end vocabulary
        self.wake_vocs = ['hello', 'music', 'create']
        self.end_vocs = ['goodbye', 'bye', 'stop']
        self.music_keywords = ['create', 'play', 'music', 'melody', 'song', 'energetic', 'peaceful', 'dramatic']
    
    def judge_voc(self, text: str) -> int:
        """Judge voice command type: 1=wake, 2=end, 3=music, 0=other"""
        text_lower = text.lower()
        
        for wake_word in self.wake_vocs:
            if wake_word in text_lower:
                return 1
                
        for end_word in self.end_vocs:
            if end_word in text_lower:
                return 2
                
        for music_word in self.music_keywords:
            if music_word in text_lower:
                return 3
                
        return 0
    
    def detect_wake(self) -> bool:
        """Detect wake word for system activation"""
        while True:
            audio_path = self.chat_handler.speech_handler.record_audio()
            text = self.chat_handler.speech_handler.recognize_speech(audio_path)
            print(f'Recognition result: {text}')
            
            voc_type = self.judge_voc(text)
            if voc_type == 1:  # Wake word detected
                self.chat_handler.speech_handler.play_audio('speech_recognition_v2/assets/di_start.mp3')
                return True
            else:
                time.sleep(0.5)
    
    def process_music_command(self, text: str) -> str:
        """Process music creation commands through VLM"""
        try:
            # Check if it's a music creation request
            if any(keyword in text.lower() for keyword in ['create', 'play', 'music']):
                print(f"Processing music command: {text}")
                
                # Use semantic anchors, constrained planning, and GMP execution
                result = self.music_creator.process_music_creation(text)
                
                # Generate speech response
                response = f"I've created and performed the music as requested. {result}"
                return response
            else:
                # Use regular chat handler for non-music commands
                return self.chat_handler.chat(text, use_history=False)
                
        except Exception as e:
            print(f"Error processing music command: {e}")
            return "Sorry, I encountered an error while creating the music."
    
    def run(self):
        """Main execution loop for the deployed Co-policy pipeline."""
        print("Co-policy Framework Started!")
        print("Say 'hello', 'music', or 'create' to wake up the system")
        
        begin_chat = False
        
        while True:
            try:
                if not begin_chat:
                    # Wait for wake word
                    if self.detect_wake():
                        text = 'Hello! I\'m ready to create music with you.'
                        begin_chat = True
                        print(text)
                        # Play greeting
                        audio_file = self.chat_handler.speech_handler.generate_audio(text)
                        self.chat_handler.speech_handler.play_audio(audio_file)
                else:
                    # Listen for user command
                    text = self.chat_handler.recognize_audio()
                    print(f"User said: {text}")
                    
                    voc_type = self.judge_voc(text)
                    
                    if voc_type == 2:  # End conversation
                        text = 'Goodbye! It was wonderful creating music together.'
                        begin_chat = False
                        print(text)
                        # Play goodbye
                        audio_file = self.chat_handler.speech_handler.generate_audio(text)
                        self.chat_handler.speech_handler.play_audio(audio_file)
                        continue
                    
                    # Process command (music or regular chat)
                    response = self.process_music_command(text)
                    print(f'System response: {response}')
                    
                    # Generate and play audio response
                    if response:
                        audio_file = self.chat_handler.speech_handler.generate_audio(response)
                        self.chat_handler.speech_handler.play_audio(audio_file)
                        
            except KeyboardInterrupt:
                print("\nShutting down Co-policy Framework...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(1)

def main():
    """Entry point for Co-policy framework"""
    framework = CoPolicyFramework()
    framework.run()

if __name__ == '__main__':
    main()