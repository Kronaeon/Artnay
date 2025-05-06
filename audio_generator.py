import os
import re
import logging
import torch
import yaml
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioGenerator:
    """
    Enhanced audio generator with Dia-1.6B model support for YouTube Shorts pipeline.
    Supports high-quality voice generation with customization options.
    """
    
    # Model Configuration
    MODEL_CONFIGS = {
        "dia_1_6b": {
            "library": "dia",
            "model_path": "/home/horus/Projects/Models/AUDIOMODELS/nari-labs",
            "voice_clone": True,
            "dialogue_support": True,
            "emotion_support": True
        },
        "openvoice_v2": {
            "library": "openvoice",
            "voice_clone": True,
            "emotion_control": True,
            "speed_control": True,
            "tone_control": True,
            "requires_voice_sample": True
        },
        "xtts_v2": {
            "library": "XTTS",
            "voice_clone": True,
            "quick_clone": True,
            "requires_voice_sample": True
        },
        "gTTS": {
            "library": "gtts",
            "voice_clone": False,
            "fallback": True
        }
    }
    
    def __init__(self, 
                 output_dir: str = "media_assets/audio",
                 model_name: str = "dia_1_6b",
                 model_path: Optional[str] = None,
                 voice_sample: Optional[str] = None,
                 device: Optional[str] = None):
        """Initialize the audio generator with specified model."""
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Model initialization
        self.current_model = model_name
        self.model = None
        self.voice_sample = voice_sample
        self.custom_model_path = model_path
        self.voice_conditioning = {}
        
        # Load the specified model
        self.load_model(model_name, model_path)
    
    def load_model(self, model_name: str, model_path: Optional[str] = None) -> None:
        """Dynamically load or switch to a different TTS model."""
        
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.MODEL_CONFIGS.keys())}")
        
        config = self.MODEL_CONFIGS[model_name]
        logging.info(f"Loading TTS model: {model_name}")
        
        # Clean up previous model if exists
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        
        try:
            # Model-specific loading logic
            if config["library"] == "dia":
                self._initialize_dia_model(model_path or config["model_path"])
                
            elif config["library"] == "openvoice":
                # OpenVoice implementation
                logging.info("OpenVoice model would be loaded here")
                self.model = "openvoice_placeholder"
                
            elif config["library"] == "XTTS":
                # XTTS implementation
                logging.info("XTTS model would be loaded here")
                self.model = "xtts_placeholder"
                
            elif config["library"] == "gtts":
                # Load gTTS as fallback
                try:
                    from gtts import gTTS
                    self.model = "gtts_fallback"
                    logging.info("Using gTTS as fallback TTS system")
                except ImportError:
                    logging.error("gTTS not installed, cannot use fallback")
                    raise
            
            self.current_model = model_name
            logging.info(f"Successfully loaded {model_name}")
            
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _initialize_dia_model(self, model_path: str) -> None:
        """Initialize the Dia model properly."""
        try:
            logging.info(f"Initializing Dia model from {model_path}")
            
            # Try to import Dia
            try:
                from dia.model import Dia
                
                # Load the model
                self.model = Dia.from_pretrained(model_path)
                
                # Optimize for inference if possible
                if hasattr(torch, 'compile') and self.device == 'cuda':
                    try:
                        self.model = torch.compile(self.model)
                        logging.info("Using torch.compile() for optimized inference")
                    except Exception as e:
                        logging.warning(f"Could not use torch.compile(): {e}")
                
                logging.info("Dia model loaded successfully")
                return True
                
            except ImportError:
                logging.error("Dia module not found. Please install with: git clone https://github.com/nari-labs/dia.git && cd dia && pip install -e .")
                raise
                
        except Exception as e:
            logging.error(f"Failed to load Dia model: {e}")
            raise
    
    def parse_script_segment(self, content: str, section_name: str) -> Tuple[str, str]:
        """
        Extract narration text and visual description from script section.
        
        Args:
            content: The full markdown content
            section_name: Section to extract (HOOK, MAIN CONTENT, CONCLUSION)
            
        Returns:
            Tuple of (visual description, narration text)
        """
        # Create regex pattern for the section
        pattern = rf'## {section_name.upper()}.*?'
        pattern += r'VISUAL:\*\* ([^\n]+)\n\n'
        pattern += r'\*\*NARRATION:\*\* "([^"]+)"'
        
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            visual = match.group(1).strip()
            narration = match.group(2).strip()
            return visual, narration
        
        return "", ""
    
    def parse_script_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from YAML front matter."""
        yaml_match = re.search(r'---\n(.*?)\n---', content, re.DOTALL)
        
        if yaml_match:
            try:
                metadata = yaml.safe_load(yaml_match.group(1))
                return metadata if metadata else {}
            except yaml.YAMLError:
                logging.warning("Failed to parse YAML metadata")
        
        return {}
    
    def generate_narration(self, 
                          script_file: str,
                          voice_settings: Optional[Dict[str, Any]] = None) -> List[Path]:
        """
        Generate audio files for all script segments.
        
        Args:
            script_file: Path to the script markdown file
            voice_settings: Optional settings for voice generation
            
        Returns:
            List of paths to generated audio files
        """
        # Read script file
        with open(script_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse metadata
        metadata = self.parse_script_metadata(content)
        logging.info(f"Processing script: {metadata.get('title', 'Unknown')}")
        
        # Extract segments
        sections = ["HOOK", "MAIN CONTENT", "CONCLUSION"]
        audio_files = []
        
        for section in sections:
            visual, narration = self.parse_script_segment(content, section)
            
            if narration:
                # Generate filename based on section
                segment_name = section.lower().replace(" ", "_")
                audio_file = self._generate_audio(segment_name, narration, voice_settings)
                
                if audio_file:
                    audio_files.append(audio_file)
                    logging.info(f"Generated audio for {section}: {audio_file}")
            else:
                logging.warning(f"No narration found for {section}")
        
        return audio_files
    
    def _generate_audio(self, 
                       segment_name: str, 
                       text: str,
                       voice_settings: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """
        Generate audio file for a single segment.
        
        Args:
            segment_name: Name of the segment (hook, main_content, conclusion)
            text: Text to convert to speech
            voice_settings: Optional settings for voice generation
            
        Returns:
            Path to generated audio file or None if failed
        """
        # Default voice settings
        settings = voice_settings or {}
        
        # Set output file path
        output_file = self.output_dir / f"{segment_name}.wav"
        
        try:
            # Model-specific generation logic
            if self.current_model == "dia_1_6b":
                return self._generate_with_dia(segment_name, text, output_file, settings)
                
            elif self.current_model == "openvoice_v2":
                # OpenVoice implementation placeholder
                logging.info(f"Would generate with OpenVoice: {text[:30]}...")
                output_file.touch()  # Create empty file as placeholder
                
            elif self.current_model == "xtts_v2":
                # XTTS implementation placeholder
                logging.info(f"Would generate with XTTS: {text[:30]}...")
                output_file.touch()  # Create empty file as placeholder
                
            elif self.current_model == "gTTS":
                return self._generate_with_gtts(segment_name, text, output_file)
            
            # Verify file was created
            if output_file.exists():
                return output_file
            else:
                logging.error(f"Failed to create audio file: {output_file}")
                return None
                                
        except Exception as e:
            logging.error(f"Error generating audio for {segment_name}: {e}")
            return None
    
    def _generate_with_dia(self, 
                          segment_name: str, 
                          text: str, 
                          output_file: Path,
                          settings: Dict[str, Any]) -> Optional[Path]:
        """Generate audio using the Dia model."""
        try:
            # Check if model is loaded
            if self.model is None or not hasattr(self.model, 'generate'):
                logging.error("Dia model not properly initialized")
                return self._generate_with_gtts(segment_name, text, output_file)
            
            # Format text for Dia
            # Apply speaker tags if voice conditioning is available
            speaker_id = settings.get('speaker_id', 'S1')
            formatted_text = f"[{speaker_id}] {text}"
            
            # Add emotion markers if specified
            emotion = settings.get('emotion')
            if emotion:
                formatted_text = f"[{speaker_id}] ({emotion}) {text}"
            
            # Apply voice conditioning if available
            voice_id = settings.get('voice_id')
            conditioning = None
            if voice_id and voice_id in self.voice_conditioning:
                conditioning = self.voice_conditioning[voice_id]
                logging.info(f"Using voice conditioning for {voice_id}")
            
            # Generate audio
            logging.info(f"Generating audio with Dia: {formatted_text[:50]}...")
            
            # Call the model's generate method
            if conditioning is not None:
                # Generate with voice conditioning
                audio_array = self.model.generate(formatted_text, conditioning=conditioning)
            else:
                # Generate with default voice
                audio_array = self.model.generate(formatted_text)
            
            # Save audio to file
            sf.write(str(output_file), audio_array, 44100)
            logging.info(f"Audio saved to {output_file}")
            
            return output_file
            
        except Exception as e:
            logging.error(f"Error generating with Dia: {e}")
            # Fall back to gTTS
            logging.info("Falling back to gTTS")
            return self._generate_with_gtts(segment_name, text, output_file)
    
    def _generate_with_gtts(self, segment_name: str, text: str, output_file: Path) -> Optional[Path]:
        """Generate audio using gTTS as fallback."""
        try:
            from gtts import gTTS
            
            logging.info(f"Generating audio for '{text[:30]}...' using gTTS")
            
            # Generate speech
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save as MP3 first (gTTS limitation)
            mp3_file = str(output_file).replace('.wav', '.mp3')
            tts.save(mp3_file)
            
            # Convert to WAV using ffmpeg if available
            try:
                subprocess.run(['ffmpeg', '-y', '-i', mp3_file, str(output_file)], 
                              check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Remove the temporary MP3 file
                os.remove(mp3_file)
            except (subprocess.SubprocessError, FileNotFoundError):
                # If ffmpeg fails or is not available, just rename the mp3
                logging.warning("ffmpeg not available, using MP3 instead of WAV")
                output_file = Path(mp3_file)
            
            logging.info(f"Created audio file: {output_file}")
            return output_file
            
        except ImportError:
            logging.error("gTTS not installed, cannot generate fallback audio")
            return None
        except Exception as e:
            logging.error(f"Error generating with gTTS: {e}")
            return None
    
    def prepare_voice_conditioning(self, 
                                  voice_sample: str, 
                                  transcript: str,
                                  voice_id: str = "custom_voice") -> bool:
        """
        Prepare voice conditioning from sample audio.
        
        Args:
            voice_sample: Path to voice sample file
            transcript: Transcript of the voice sample
            voice_id: Identifier for this voice
            
        Returns:
            True if successful, False otherwise
        """
        if not Path(voice_sample).exists():
            logging.error(f"Voice sample not found: {voice_sample}")
            return False
        
        try:
            # Check if current model supports voice cloning
            if self.current_model != "dia_1_6b":
                logging.warning(f"Voice conditioning only supported with Dia model")
                return False
            
            logging.info(f"Preparing voice conditioning from {voice_sample}")
            
            # Load audio sample
            audio, sr = sf.read(voice_sample)
            
            # Ensure audio is mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Ensure sample rate is correct (Dia expects 44.1kHz)
            if sr != 44100:
                logging.warning(f"Resampling audio from {sr}Hz to 44100Hz")
                # This would need a resampling implementation
            
            # Generate conditioning with Dia
            # formatted_transcript should have proper speaker tags
            formatted_transcript = f"[S1] {transcript}"
            
            # Extract conditioning
            conditioning = self.model.extract_conditioning(
                audio_path=voice_sample,
                transcript=formatted_transcript
            )
            
            # Store conditioning for later use
            self.voice_conditioning[voice_id] = conditioning
            
            logging.info(f"Voice conditioning for {voice_id} created successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error preparing voice conditioning: {e}")
            return False
    
    def blend_voices(self, 
                    voice_id1: str, 
                    voice_id2: str, 
                    blend_ratio: float = 0.5,
                    blended_voice_id: str = "blended") -> bool:
        """
        Blend two voice conditionings.
        
        Args:
            voice_id1: First voice ID
            voice_id2: Second voice ID
            blend_ratio: Blending ratio (0.0 to 1.0, where 0.0 is all voice1, 1.0 is all voice2)
            blended_voice_id: ID for the blended voice
            
        Returns:
            True if successful, False otherwise
        """
        if voice_id1 not in self.voice_conditioning or voice_id2 not in self.voice_conditioning:
            missing = []
            if voice_id1 not in self.voice_conditioning:
                missing.append(voice_id1)
            if voice_id2 not in self.voice_conditioning:
                missing.append(voice_id2)
            logging.error(f"Voice conditioning not found for: {', '.join(missing)}")
            return False
        
        try:
            # Get conditioning vectors
            cond1 = self.voice_conditioning[voice_id1]
            cond2 = self.voice_conditioning[voice_id2]
            
            # Blend conditioning vectors
            # Note: This is a simplified approach - actual implementation depends on Dia's conditioning format
            if isinstance(cond1, dict) and isinstance(cond2, dict):
                # If conditioning is a dictionary of tensors
                blended_cond = {}
                for key in cond1.keys():
                    if key in cond2 and torch.is_tensor(cond1[key]) and torch.is_tensor(cond2[key]):
                        blended_cond[key] = (1 - blend_ratio) * cond1[key] + blend_ratio * cond2[key]
                    else:
                        blended_cond[key] = cond1[key]  # Use first conditioning if key not in both
            elif torch.is_tensor(cond1) and torch.is_tensor(cond2):
                # If conditioning is a single tensor
                blended_cond = (1 - blend_ratio) * cond1 + blend_ratio * cond2
            else:
                logging.error("Unsupported conditioning format for blending")
                return False
            
            # Store blended conditioning
            self.voice_conditioning[blended_voice_id] = blended_cond
            
            logging.info(f"Successfully blended voices into {blended_voice_id} (ratio: {blend_ratio:.2f})")
            return True
            
        except Exception as e:
            logging.error(f"Error blending voices: {e}")
            return False
    
    @staticmethod
    def list_available_models() -> List[str]:
        """List all available TTS models."""
        return list(AudioGenerator.MODEL_CONFIGS.keys())
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """Get information about the current or specified model."""
        model_name = model_name or self.current_model
        
        if model_name not in self.MODEL_CONFIGS:
            return {}
        
        return self.MODEL_CONFIGS[model_name]
    
    def estimate_audio_duration(self, text: str, wpm: int = 150) -> float:
        """
        Estimate audio duration based on text length.
        
        Args:
            text: Text to estimate
            wpm: Words per minute (default: 150)
            
        Returns:
            Estimated duration in seconds
        """
        word_count = len(text.split())
        duration = (word_count / wpm) * 60
        return duration


# Example usage for testing
if __name__ == "__main__":
    # Test the audio generator
    try:
        generator = AudioGenerator(model_name="dia_1_6b")
        
        # Test with a simple sentence
        test_text = "This is a test of the Dia voice generation system."
        print(f"Generating audio for: {test_text}")
        
        test_file = generator._generate_audio("test", test_text, {})
        print(f"Generated file: {test_file}")
        
    except Exception as e:
        print(f"Error: {e}")