"""
Voice Handler for Ponniyin Selvan Chatbot
ElevenLabs Text-to-Speech integration
"""
import os
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class VoiceHandler:
    """ElevenLabs TTS integration for voice output"""
    
    # Default voices
    DEFAULT_VOICE_EN = "Adam"  # Clear, professional English voice
    DEFAULT_VOICE_TA = "Adam"  # Using English voice for Tamil (ElevenLabs has limited Tamil support)
    
    # Voice IDs can be customized
    VOICE_OPTIONS = {
        "Adam": "pNInz6obpgDQGcFmaJgB",      # Deep, narrative voice
        "Antoni": "ErXwobaYiN019PkySvjV",    # Warm, friendly
        "Arnold": "VR6AewLTigWG4xSOukaG",    # Authoritative
        "Rachel": "21m00Tcm4TlvDq8ikWAM",    # Female, clear
        "Domi": "AZnzlk1XvdvUeBnXmlld",      # Female, expressive
    }
    
    def __init__(self, voice_en: str = None, voice_ta: str = None):
        """
        Initialize the voice handler
        
        Args:
            voice_en: Voice name for English responses
            voice_ta: Voice name for Tamil responses
        """
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.client = None
        self.voice_en = voice_en or self.DEFAULT_VOICE_EN
        self.voice_ta = voice_ta or self.DEFAULT_VOICE_TA
        self._init_client()
    
    def _init_client(self):
        """Initialize the ElevenLabs client"""
        if not self.api_key:
            print("Warning: ELEVENLABS_API_KEY not found. Voice features disabled.")
            return
        
        try:
            from elevenlabs import ElevenLabs
            self.client = ElevenLabs(api_key=self.api_key)
        except ImportError:
            print("Warning: elevenlabs package not installed. Run: pip install elevenlabs")
        except Exception as e:
            print(f"Warning: Could not initialize ElevenLabs client: {e}")
    
    def _get_voice_id(self, voice_name: str) -> str:
        """Get voice ID from voice name"""
        return self.VOICE_OPTIONS.get(voice_name, self.VOICE_OPTIONS["Adam"])
    
    def text_to_speech(
        self, 
        text: str, 
        language: str = "en"
    ) -> Optional[bytes]:
        """
        Convert text to speech
        
        Args:
            text: Text to convert
            language: 'en' for English, 'ta' for Tamil
            
        Returns:
            Audio bytes or None if failed
        """
        if not self.client:
            return None
        
        # Select voice based on language
        voice_name = self.voice_ta if language == 'ta' else self.voice_en
        voice_id = self._get_voice_id(voice_name)
        
        try:
            # Generate audio
            audio_generator = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_multilingual_v2",  # Best for multilingual including Tamil
                output_format="mp3_44100_128"
            )
            
            # Collect audio bytes from generator
            audio_bytes = b"".join(audio_generator)
            return audio_bytes
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None
    
    def text_to_speech_file(
        self, 
        text: str, 
        output_path: str,
        language: str = "en"
    ) -> bool:
        """
        Convert text to speech and save to file
        
        Args:
            text: Text to convert
            output_path: Path to save the audio file
            language: 'en' for English, 'ta' for Tamil
            
        Returns:
            True if successful, False otherwise
        """
        audio_bytes = self.text_to_speech(text, language)
        
        if audio_bytes:
            try:
                with open(output_path, 'wb') as f:
                    f.write(audio_bytes)
                return True
            except Exception as e:
                print(f"Error saving audio file: {e}")
        
        return False
    
    def get_available_voices(self) -> list:
        """Get list of available voice names"""
        return list(self.VOICE_OPTIONS.keys())
    
    def is_available(self) -> bool:
        """Check if voice features are available"""
        return self.client is not None


# Global instance for convenience
_voice_handler = None

def get_voice_handler() -> VoiceHandler:
    """Get or create the global voice handler instance"""
    global _voice_handler
    if _voice_handler is None:
        _voice_handler = VoiceHandler()
    return _voice_handler


if __name__ == "__main__":
    # Test voice handler
    handler = VoiceHandler()
    
    if handler.is_available():
        print("Voice handler initialized successfully!")
        print(f"Available voices: {handler.get_available_voices()}")
        
        # Test English TTS
        success = handler.text_to_speech_file(
            "Welcome to Ponniyin Selvan, the epic tale of the Chola dynasty.",
            "test_en.mp3",
            language="en"
        )
        print(f"English TTS test: {'Success' if success else 'Failed'}")
        
        # Test Tamil TTS
        success = handler.text_to_speech_file(
            "பொன்னியின் செல்வன் வரலாற்று நாவலுக்கு வரவேற்கிறோம்.",
            "test_ta.mp3",
            language="ta"
        )
        print(f"Tamil TTS test: {'Success' if success else 'Failed'}")
    else:
        print("Voice handler not available. Check API key.")
