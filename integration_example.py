"""
Enhanced YouTube Shorts Pipeline Integration with Dia Voice Generation
This script combines content generation, script creation, and advanced audio generation.
"""

import os
import sys
import logging
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import datetime

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline components
from tubeShortsScriptGen import YouTubeShortsScriptGenerator
from audio_generator import AudioGenerator
from voice_preparation import VoicePreparation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_shorts_pipeline.log'),
        logging.StreamHandler()
    ]
)

class EnhancedYouTubeShortsCreator:
    """
    Enhanced YouTube Shorts content creation pipeline with advanced voice generation.
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 input_dir: str = "CleanSC",
                 output_dir: str = "media_assets",
                 use_dia: bool = True):
        """Initialize the pipeline with configuration."""
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = self._load_config(config_file)
        self.use_dia = use_dia
        
        # Create necessary directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / "scripts").mkdir(exist_ok=True)
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)
        (self.output_dir / "voice_samples").mkdir(exist_ok=True)
        
        # Initialize components
        self.script_generator = None
        self.audio_generator = None
        self.voice_preparation = None
        
        # Voice settings
        self.voices = {}
        self.blended_voices = {}
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "script_model": {
                "model_path": "/home/horus/Projects/Models/gemma-3-finetune.Q8_0.gguf",
                "n_gpu_layers": 35,
                "n_ctx": 2048,
                "n_batch": 512
            },
            "audio_model": {
                "model_name": "dia_1_6b",
                "model_path": "/home/horus/Projects/Models/AUDIOMODELS/nari-labs",
                "voice_samples_dir": "voice_samples"
            },
            "voice_settings": {
                "default": {
                    "speaker_id": "S1",
                    "emotion": "friendly"
                }
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge configs
                for section, values in loaded_config.items():
                    if section in default_config and isinstance(default_config[section], dict):
                        default_config[section].update(values)
                    else:
                        default_config[section] = values
                logging.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logging.error(f"Error loading config file: {e}")
        
        return default_config
    
    def initialize(self):
        """Initialize all pipeline components."""
        self._initialize_script_generator()
        self._initialize_audio_generator()
        self._initialize_voice_preparation()
    
    def _initialize_script_generator(self):
        """Initialize the script generation component."""
        logging.info("Initializing script generator...")
        
        config = self.config["script_model"]
        self.script_generator = YouTubeShortsScriptGenerator(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir / "scripts"),
            model_path=config["model_path"],
            n_gpu_layers=config["n_gpu_layers"],
            n_ctx=config["n_ctx"],
            n_batch=config["n_batch"]
        )
    
    def _initialize_audio_generator(self):
        """Initialize the audio generation component."""
        logging.info("Initializing audio generator...")
        
        config = self.config["audio_model"]
        model_name = "dia_1_6b" if self.use_dia else config.get("model_name", "gTTS")
        
        self.audio_generator = AudioGenerator(
            output_dir=str(self.output_dir / "audio"),
            model_name=model_name,
            model_path=config.get("model_path")
        )
    
    def _initialize_voice_preparation(self):
        """Initialize voice preparation utilities."""
        logging.info("Initializing voice preparation utilities...")
        
        self.voice_preparation = VoicePreparation(
            output_dir=str(self.output_dir / "voice_samples")
        )
    
    def create_script(self, topic: Optional[str] = None, style: str = "informative") -> Dict[str, Any]:
        """
        Generate a script for a YouTube Short.
        
        Args:
            topic: Optional specific topic (otherwise uses cleaned content)
            style: Script style ("informative", "entertaining", "educational")
            
        Returns:
            Dictionary with script information
        """
        logging.info(f"Generating {style} script for topic: {topic or 'from cleaned content'}")
        
        try:
            script_data = self.script_generator.generate_script(style=style)
            script_file = script_data["file_path"]
            
            logging.info(f"Script generated successfully: {script_data['title']}")
            return {
                "success": True,
                "script_data": script_data,
                "script_file": script_file
            }
        except Exception as e:
            logging.error(f"Script generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def prepare_voice(self, 
                     voice_sample: str, 
                     transcript: str,
                     voice_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare a voice sample for use in narration.
        
        Args:
            voice_sample: Path to voice sample file
            transcript: Transcript of the sample
            voice_id: ID for this voice
            
        Returns:
            Dictionary with voice preparation result
        """
        logging.info(f"Preparing voice sample: {voice_sample}")
        
        try:
            # Prepare training sample
            result = self.voice_preparation.prepare_training_sample(
                voice_sample, transcript, voice_id)
            
            if not result["success"]:
                return result
            
            voice_id = result["voice_id"]
            
            # Create voice conditioning
            if self.use_dia:
                success = self.audio_generator.prepare_voice_conditioning(
                    voice_sample, transcript, voice_id)
                
                if success:
                    # Store voice information
                    self.voices[voice_id] = {
                        "sample": voice_sample,
                        "transcript": transcript,
                        "prepared": True
                    }
                    
                    logging.info(f"Voice {voice_id} prepared successfully")
                    return {
                        "success": True,
                        "voice_id": voice_id,
                        "message": f"Voice {voice_id} prepared successfully"
                    }
                else:
                    logging.error(f"Voice conditioning failed for {voice_id}")
                    return {
                        "success": False,
                        "error": f"Voice conditioning failed for {voice_id}"
                    }
            else:
                logging.warning("Voice conditioning not available without Dia model")
                return {
                    "success": False,
                    "error": "Voice conditioning requires Dia model"
                }
                
        except Exception as e:
            logging.error(f"Voice preparation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def blend_voices(self, 
                    voice_id1: str, 
                    voice_id2: str, 
                    blend_ratio: float = 0.5,
                    blended_voice_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Blend two voices into a new voice.
        
        Args:
            voice_id1: First voice ID
            voice_id2: Second voice ID
            blend_ratio: Blending ratio (0.0 to 1.0, where 0.0 is all voice1, 1.0 is all voice2)
            blended_voice_id: ID for the blended voice
            
        Returns:
            Dictionary with blending result
        """
        logging.info(f"Blending voices {voice_id1} and {voice_id2} (ratio: {blend_ratio:.2f})")
        
        if not self.use_dia:
            return {
                "success": False,
                "error": "Voice blending requires Dia model"
            }
        
        try:
            # Check if voices exist
            if voice_id1 not in self.voices:
                return {
                    "success": False,
                    "error": f"Voice {voice_id1} not found"
                }
            
            if voice_id2 not in self.voices:
                return {
                    "success": False,
                    "error": f"Voice {voice_id2} not found"
                }
            
            # Generate blended voice ID if not provided
            if not blended_voice_id:
                timestamp = datetime.datetime.now().strftime("%m%d%H%M")
                blended_voice_id = f"blend_{voice_id1}_{voice_id2}_{timestamp}"
            
            # Blend voices
            success = self.audio_generator.blend_voices(
                voice_id1, voice_id2, blend_ratio, blended_voice_id)
            
            if success:
                # Store blended voice information
                self.blended_voices[blended_voice_id] = {
                    "voice1": voice_id1,
                    "voice2": voice_id2,
                    "blend_ratio": blend_ratio
                }
                
                logging.info(f"Voices blended successfully into {blended_voice_id}")
                return {
                    "success": True,
                    "blended_voice_id": blended_voice_id,
                    "message": f"Voices blended successfully into {blended_voice_id}"
                }
            else:
                logging.error(f"Voice blending failed")
                return {
                    "success": False,
                    "error": "Voice blending failed"
                }
                
        except Exception as e:
            logging.error(f"Voice blending failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_narration(self, 
                         script_file: str,
                         voice_id: Optional[str] = None,
                         custom_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate audio narration for a script.
        
        Args:
            script_file: Path to script file
            voice_id: ID of voice to use
            custom_settings: Additional voice settings
            
        Returns:
            Dictionary with narration result
        """
        logging.info(f"Generating narration for script: {script_file}")
        
        try:
            # Prepare voice settings
            voice_settings = dict(self.config["voice_settings"].get("default", {}))
            
            if voice_id:
                voice_settings["voice_id"] = voice_id
            
            if custom_settings:
                voice_settings.update(custom_settings)
            
            # Generate narration
            audio_files = self.audio_generator.generate_narration(
                script_file=script_file,
                voice_settings=voice_settings
            )
            
            if audio_files:
                logging.info(f"Narration generated successfully: {len(audio_files)} audio files")
                return {
                    "success": True,
                    "audio_files": [str(file) for file in audio_files],
                    "voice_settings": voice_settings
                }
            else:
                logging.error("No audio files generated")
                return {
                    "success": False,
                    "error": "No audio files generated"
                }
                
        except Exception as e:
            logging.error(f"Narration generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_complete_short(self, 
                            style: str = "informative",
                            voice_id: Optional[str] = None,
                            emotion: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a complete YouTube Short from script to narration.
        
        Args:
            style: Script style
            voice_id: Voice ID to use
            emotion: Emotion for narration
            
        Returns:
            Dictionary with creation result
        """
        logging.info(f"Creating complete YouTube Short (style: {style})")
        
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "style": style,
            "voice_id": voice_id,
            "emotion": emotion,
            "steps": {}
        }
        
        try:
            # Step 1: Generate script
            script_result = self.create_script(style=style)
            results["steps"]["script"] = script_result
            
            if not script_result["success"]:
                results["success"] = False
                results["error"] = f"Script generation failed: {script_result.get('error')}"
                return results
            
            script_file = script_result["script_file"]
            
            # Step 2: Generate narration
            voice_settings = {}
            if voice_id:
                voice_settings["voice_id"] = voice_id
            if emotion:
                voice_settings["emotion"] = emotion
            
            narration_result = self.generate_narration(script_file, voice_id, voice_settings)
            results["steps"]["narration"] = narration_result
            
            if not narration_result["success"]:
                results["success"] = False
                results["error"] = f"Narration generation failed: {narration_result.get('error')}"
                return results
            
            # Return successful result
            results["success"] = True
            results["message"] = "YouTube Short created successfully"
            return results
            
        except Exception as e:
            logging.error(f"YouTube Short creation failed: {e}")
            results["success"] = False
            results["error"] = str(e)
            return results
    
    def download_and_prepare_youtube_voice(self, 
                                         youtube_url: str,
                                         transcript: str,
                                         voice_id: Optional[str] = None,
                                         start_time: Optional[str] = None,
                                         end_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Download a YouTube video and prepare voice from it.
        
        Args:
            youtube_url: YouTube video URL
            transcript: Transcript of the voice sample
            voice_id: ID for this voice
            start_time: Start time for trimming (MM:SS)
            end_time: End time for trimming (MM:SS)
            
        Returns:
            Dictionary with preparation result
        """
        logging.info(f"Downloading and preparing voice from YouTube: {youtube_url}")
        
        try:
            # Generate voice ID if not provided
            if not voice_id:
                # Extract video ID from URL
                import re
                video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
                if video_id:
                    voice_id = f"yt_{video_id.group(1)}"
                else:
                    timestamp = datetime.datetime.now().strftime("%m%d%H%M")
                    voice_id = f"youtube_{timestamp}"
            
            # Download audio
            audio_file = self.voice_preparation.download_youtube_audio(
                youtube_url, voice_id, start_time, end_time)
            
            if not audio_file:
                return {
                    "success": False,
                    "error": "Failed to download audio from YouTube"
                }
            
            # Prepare voice
            return self.prepare_voice(audio_file, transcript, voice_id)
            
        except Exception as e:
            logging.error(f"YouTube voice preparation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Enhanced YouTube Shorts Creator")
    
    # Configuration
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--input-dir", default="CleanSC", help="Input directory with cleaned content")
    parser.add_argument("--output-dir", default="media_assets", help="Output directory for media assets")
    parser.add_argument("--disable-dia", action="store_true", help="Disable Dia model and use fallback")
    
    # Commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create script
    script_parser = subparsers.add_parser("script", help="Generate a script")
    script_parser.add_argument("--style", default="informative", 
                             choices=["informative", "entertaining", "educational", "controversial"],
                             help="Script style")
    
    # Prepare voice
    voice_parser = subparsers.add_parser("voice", help="Prepare a voice sample")
    voice_parser.add_argument("sample", help="Path to voice sample file")
    voice_parser.add_argument("transcript", help="Transcript of the sample")
    voice_parser.add_argument("--id", help="Voice ID")
    
    # Download YouTube voice
    youtube_parser = subparsers.add_parser("youtube", help="Download and prepare YouTube voice")
    youtube_parser.add_argument("url", help="YouTube URL")
    youtube_parser.add_argument("transcript", help="Transcript of the voice sample")
    youtube_parser.add_argument("--id", help="Voice ID")
    youtube_parser.add_argument("--start", help="Start time (MM:SS)")
    youtube_parser.add_argument("--end", help="End time (MM:SS)")
    
    # Blend voices
    blend_parser = subparsers.add_parser("blend", help="Blend two voices")
    blend_parser.add_argument("voice1", help="First voice ID")
    blend_parser.add_argument("voice2", help="Second voice ID")
    blend_parser.add_argument("--ratio", type=float, default=0.5, help="Blend ratio (0.0-1.0)")
    blend_parser.add_argument("--id", help="Blended voice ID")
    
    # Generate narration
    narration_parser = subparsers.add_parser("narration", help="Generate narration for a script")
    narration_parser.add_argument("script", help="Path to script file")
    narration_parser.add_argument("--voice", help="Voice ID to use")
    narration_parser.add_argument("--emotion", help="Emotion for narration")
    
    # Create complete short
    short_parser = subparsers.add_parser("short", help="Create a complete YouTube Short")
    short_parser.add_argument("--style", default="informative", 
                            choices=["informative", "entertaining", "educational", "controversial"],
                            help="Script style")
    short_parser.add_argument("--voice", help="Voice ID to use")
    short_parser.add_argument("--emotion", help="Emotion for narration")
    
    args = parser.parse_args()
    
    # Initialize creator
    creator = EnhancedYouTubeShortsCreator(
        config_file=args.config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_dia=not args.disable_dia
    )
    
    # Initialize components
    creator.initialize()
    
    # Execute command
    result = None
    
    if args.command == "script":
        result = creator.create_script(style=args.style)
        
    elif args.command == "voice":
        result = creator.prepare_voice(args.sample, args.transcript, args.id)
        
    elif args.command == "youtube":
        result = creator.download_and_prepare_youtube_voice(
            args.url, args.transcript, args.id, args.start, args.end)
        
    elif args.command == "blend":
        result = creator.blend_voices(args.voice1, args.voice2, args.ratio, args.id)
        
    elif args.command == "narration":
        result = creator.generate_narration(args.script, args.voice, 
                                         {"emotion": args.emotion} if args.emotion else None)
        
    elif args.command == "short":
        result = creator.create_complete_short(args.style, args.voice, args.emotion)
        
    else:
        parser.print_help()
        return
    
    # Print result
    if result:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
