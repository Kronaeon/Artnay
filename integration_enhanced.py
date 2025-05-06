"""
Enhanced YouTube Shorts Pipeline Integration with Dia Voice Cloning
This script combines content generation, script creation, and high-quality voice cloning
with proper GPU memory management.
"""

import os
import sys
import logging
import json
import argparse
import re
import subprocess
import gc
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
import datetime
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_shorts_pipeline.log'),
        logging.StreamHandler()
    ]
)

@contextmanager
def gpu_memory_manager():
    """Context manager to ensure proper GPU memory cleanup between operations."""
    try:
        yield
    finally:
        # Force garbage collection
        gc.collect()
        
        # Clean CUDA cache if torch is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info(f"GPU memory cleared. Available: {torch.cuda.memory_allocated(0)/1e9:.2f}GB used, {torch.cuda.memory_reserved(0)/1e9:.2f}GB reserved")
        
class EnhancedYouTubeShortsCreator:
    """
    Enhanced YouTube Shorts content creation pipeline with high-quality voice cloning
    and memory management.
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 input_dir: str = "CleanSC",
                 output_dir: str = "media_assets",
                 use_dia: bool = True):
        """Initialize the pipeline with configuration."""
        
        self.input_dir = Path(input_dir or "CleanSC")
        self.output_dir = Path(output_dir or "media_assets")
        self.config = self._load_config(config_file)
        self.use_dia = use_dia
        
        # Create necessary directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / "scripts").mkdir(exist_ok=True)
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)
        (self.output_dir / "voice_samples").mkdir(exist_ok=True)
        
        # Initialize references to components (load only when needed)
        self.script_generator = None
        self.voice_preparation = None
        
        # Import components (classes only, not instances)
        try:
            # Import but don't instantiate
            from tubeShortsScriptGen import YouTubeShortsScriptGenerator
            from voice_preparation import VoicePreparation
            
            self.YouTubeShortsScriptGenerator = YouTubeShortsScriptGenerator
            self.VoicePreparation = VoicePreparation
        except ImportError as e:
            logging.error(f"Failed to import required modules: {e}")
            logging.error("Make sure tubeShortsScriptGen.py and voice_preparation.py are in your Python path.")
            raise
    
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
        """Initialize needed pipeline components."""
        # Initialize voice preparation (minimal memory usage)
        self._initialize_voice_preparation()
        # Script generator is only initialized when needed
    
    def _initialize_script_generator(self):
        """Initialize the script generation component when needed."""
        if self.script_generator is not None:
            logging.info("Script generator already initialized")
            return
            
        logging.info("Initializing script generator...")
        
        with gpu_memory_manager():
            try:
                config = self.config["script_model"]
                self.script_generator = self.YouTubeShortsScriptGenerator(
                    input_dir=str(self.input_dir),
                    output_dir=str(self.output_dir / "scripts"),
                    model_path=config["model_path"],
                    n_gpu_layers=config["n_gpu_layers"],
                    n_ctx=config["n_ctx"],
                    n_batch=config["n_batch"]
                )
            except Exception as e:
                logging.error(f"Failed to initialize script generator: {e}")
                raise
    
    def _unload_script_generator(self):
        """Explicitly unload the script generator to free GPU memory."""
        if self.script_generator is not None:
            logging.info("Unloading script generator to free GPU memory")
            
            # Set to None to allow garbage collection
            self.script_generator = None
            
            # Force garbage collection and clear CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info(f"GPU memory after unloading: {torch.cuda.memory_allocated(0)/1e9:.2f}GB used, {torch.cuda.memory_reserved(0)/1e9:.2f}GB reserved")
    
    def _initialize_voice_preparation(self):
        """Initialize voice preparation utilities."""
        logging.info("Initializing voice preparation utilities...")
        
        try:
            self.voice_preparation = self.VoicePreparation(
                output_dir=str(self.output_dir / "voice_samples")
            )
        except Exception as e:
            logging.error(f"Failed to initialize voice preparation: {e}")
            raise
    
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
            # Initialize script generator if needed
            self._initialize_script_generator()
            
            # Generate the script
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
    
    def generate_narration_with_clone(self, 
                                    script_file: str,
                                    voice_sample: str,
                                    voice_transcript: str,
                                    output_dir: Optional[str] = None,
                                    voice_settings: Optional[Dict[str, Any]] = None,
                                    create_variants: bool = False,
                                    preprocess: bool = False) -> Dict[str, Any]:
        """Generate audio narration using voice cloning technology."""
        logging.info(f"Generating narration with voice cloning for script: {script_file}")
        
        try:
            # IMPROVED PATH RESOLUTION - avoid path doubling
            script_path = Path(script_file)
            if not script_path.is_absolute() and not "media_assets/scripts" in str(script_path):
                script_path = self.output_dir / "scripts" / script_path.name
            
            voice_sample_path = Path(voice_sample)
            if not voice_sample_path.is_absolute() and not "media_assets/voice_samples" in str(voice_sample_path):
                voice_sample_path = self.output_dir / "voice_samples" / voice_sample_path.name
            
            # Handle voice transcript similarly
            voice_transcript_path = Path(voice_transcript)
            if not voice_transcript_path.is_absolute() and not "media_assets/voice_samples" in str(voice_transcript_path):
                voice_transcript_path = self.output_dir / "voice_samples" / voice_transcript_path.name

            logging.info(f"Resolved paths:")
            logging.info(f"- Script: {script_path}")
            logging.info(f"- Voice sample: {voice_sample_path}")
            logging.info(f"- Voice transcript: {voice_transcript_path}")
            
            # Check if script file exists
            if not script_path.exists():
                logging.error(f"Script file not found: {script_path}")
                return {"success": False, "error": f"Script file not found: {script_path}"}
            
            # Create output directory
            if output_dir is None:
                output_dir = str(self.output_dir / "audio" / script_path.stem)
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract narration text from script - USING RESOLVED PATH
            with open(script_path, 'r') as f:
                script_content = f.read()
            
            # DEBUG: Print the beginning of script content to verify
            logging.info(f"Script content preview: {script_content[:200]}...")            
            
            def extract_narration(content, section_name):
                # Create pattern to match markdown-formatted headings
                pattern = fr'## {section_name}\s+\*\*VISUAL:\*\*.*?\*\*NARRATION:\*\*\s*"([^"]+)"'
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    return match.group(1)
                return None

            # Extract narration from each section
            hook_text = extract_narration(script_content, "HOOK")
            main_text = extract_narration(script_content, "MAIN CONTENT")
            conclusion_text = extract_narration(script_content, "CONCLUSION")

            # Log what we found
            if hook_text:
                logging.info(f"Found hook text: {hook_text[:30]}...")
            if main_text:
                logging.info(f"Found main content text: {main_text[:30]}...")
            if conclusion_text:
                logging.info(f"Found conclusion text: {conclusion_text[:30]}...")

            # Combine into test text for voice cloning
            test_text = ""
            if hook_text:
                test_text += hook_text + " "
            if main_text:
                test_text += main_text + " "
            if conclusion_text:
                test_text += conclusion_text

            if not test_text:
                # Debug more detailed script structure if no narration found
                logging.error("No narration text found in script")
                logging.info("Script sections:")
                sections = script_content.split("##")
                for i, section in enumerate(sections):
                    if i > 0:  # Skip first split which is before any ##
                        logging.info(f"Section {i}: {section[:100]}...")
                return {"success": False, "error": "No narration text found in script"}
            
            # Check if voice_transcript_path exists (i.e., user passed a filename in voice_samples/)
            transcript_content = voice_transcript
            if voice_transcript_path.exists():
                with open(voice_transcript_path, 'r') as f:
                    transcript_content = f.read().strip()
                transcript_file = str(voice_transcript_path)
            else:
                # If it's not a file, assume it's raw text and save to temp file in output dir
                transcript_file = os.path.join(output_dir, "temp_transcript.txt")
                with open(transcript_file, 'w') as f:
                    f.write(transcript_content)

            # IMPORTANT: Unload script generator to free GPU memory before voice cloning
            self._unload_script_generator()
            
            # Wait for GPU memory to be released
            with gpu_memory_manager():
                logging.info("GPU memory cleared for voice cloning")
            
            # Prepare voice_clone.py command
            cmd = ["python", "voice_clone.py", str(voice_sample_path), transcript_file, 
                "--output-dir", output_dir, "--test-text", test_text]
            
            # Add optional parameters
            if voice_settings:
                if "pitch" in voice_settings:
                    cmd.extend(["--pitch", str(voice_settings["pitch"])])
                if "speed" in voice_settings:
                    cmd.extend(["--speed", str(voice_settings["speed"])])
                if "brightness" in voice_settings:
                    cmd.extend(["--brightness", str(voice_settings["brightness"])])
                if "expressiveness" in voice_settings:
                    cmd.extend(["--expressiveness", str(voice_settings["expressiveness"])])
            
            if create_variants:
                cmd.append("--create-variants")
            
            if preprocess:
                cmd.append("--preprocess")
            
            # Get model path from config if not specified in voice_settings
            model_path = self.config["audio_model"].get("model_path")
            if model_path:
                cmd.extend(["--model-path", model_path])
            
            # Set environment variable for PyTorch to avoid fragmentation
            env = os.environ.copy()
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            # Run voice_clone.py
            logging.info(f"Running voice cloning with command: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, 
                                      capture_output=True, 
                                      text=True, 
                                      check=True,
                                      env=env)
                logging.info(f"Voice cloning process output: {result.stdout}")
                if result.stderr:
                    logging.warning(f"Voice cloning process warnings: {result.stderr}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Voice cloning failed: {e.stderr}")
                return {
                    "success": False,
                    "error": f"Voice cloning failed: {e.stderr}"
                }
            
            # Process results - look for generated audio files
            audio_files = list(Path(output_dir).glob("*.wav"))
            if not audio_files:
                logging.warning(f"No audio files found in {output_dir} after voice cloning")
            
            # Create segment mapping for the main segments
            segment_audio_files = self._map_segments_to_audio_files(script_file, audio_files)
            
            return {
                "success": True,
                "output_dir": output_dir,
                "audio_files": [str(file) for file in audio_files],
                "segment_audio_files": segment_audio_files
            }
            
        except Exception as e:
            logging.error(f"Narration generation with voice cloning failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _map_segments_to_audio_files(self, script_file, audio_files):
        """
        Map script segments to appropriate audio files.
        Utilizes the cloned voice files for hook, main content, and conclusion.
        """
        # Default mapping using cloned_modified.wav or main cloned output
        main_audio = next((f for f in audio_files if "cloned_modified" in f.name), None)
        if not main_audio and audio_files:
            main_audio = audio_files[0]  # Use first audio file if no modified clone found
        
        # If no audio files found, return empty mapping
        if not main_audio:
            return {}
            
        # For now, use the same file for all segments
        segment_mapping = {
            "hook": str(main_audio),
            "main_content": str(main_audio),
            "conclusion": str(main_audio)
        }
        
        # Look for variant files that might match segment needs
        variant_mappings = {
            "hook": ["variant_faster", "variant_higher_pitch", "variant_expressive"],
            "main_content": ["cloned_modified", "variant_combined_1"],
            "conclusion": ["variant_slower", "variant_combined_2"]
        }
        
        # Try to find better matches for each segment
        for segment, variants in variant_mappings.items():
            for variant in variants:
                variant_file = next((f for f in audio_files if variant in f.name), None)
                if variant_file:
                    segment_mapping[segment] = str(variant_file)
                    break
                    
        return segment_mapping
    
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
            
            # Return the downloaded audio file info
            return {
                "success": True,
                "voice_id": voice_id,
                "audio_file": str(audio_file),
                "transcript": transcript,
                "message": f"YouTube audio downloaded successfully as {voice_id}"
            }
            
        except Exception as e:
            logging.error(f"YouTube voice preparation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def create_complete_short_with_clone(self, 
                                       style: str = "informative",
                                       voice_sample: str = None,
                                       voice_transcript: str = None,
                                       voice_settings: Optional[Dict[str, Any]] = None,
                                       create_variants: bool = False,
                                       preprocess: bool = True) -> Dict[str, Any]:
        """
        Create a complete YouTube Short with voice cloning.
        
        Args:
            style: Script style
            voice_sample: Path to voice sample file
            voice_transcript: Path to voice transcript file
            voice_settings: Voice settings (pitch, speed, etc.)
            create_variants: Whether to create voice variants
            preprocess: Whether to preprocess the voice sample
            
        Returns:
            Dictionary with creation result
        """
        logging.info(f"Creating complete YouTube Short with voice cloning (style: {style})")
        
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "style": style,
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
            
            # IMPORTANT: Unload script generator to free memory for voice_clone
            self._unload_script_generator()
            
            # Step 2: Generate narration with voice cloning
            narration_result = self.generate_narration_with_clone(
                script_file, 
                voice_sample, 
                voice_transcript,
                voice_settings=voice_settings,
                create_variants=create_variants,
                preprocess=preprocess
            )
            results["steps"]["narration"] = narration_result
            
            if not narration_result["success"]:
                results["success"] = False
                results["error"] = f"Narration generation failed: {narration_result.get('error')}"
                return results
            
            # Return successful result
            results["success"] = True
            results["message"] = "YouTube Short created successfully with voice cloning"
            return results
            
        except Exception as e:
            logging.error(f"YouTube Short creation with voice cloning failed: {e}")
            results["success"] = False
            results["error"] = str(e)
            return results


    def _resolve_path(self, path_str: str, type_dir: str) -> Path:
        """Helper to resolve a path relative to the expected directory structure.
        
        Args:
            path_str: The path string from CLI argument
            type_dir: The expected subdirectory ('scripts', 'voice_samples', etc.)
            
        Returns:
            Path: The resolved path
        """
        path = Path(path_str)
        if not path.is_absolute() and not (path.exists() and path.is_file()):
            # Try resolving relative to expected directory
            resolved_path = self.output_dir / type_dir / path_str
            if resolved_path.exists():
                return resolved_path
                
        # Either the path is absolute, or it's already a file, or we need to trust user input
        return path

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Enhanced YouTube Shorts Creator with Voice Cloning")
    
    # Configuration
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--input-dir", default="CleanSC", help="Input directory with cleaned content")
    parser.add_argument("--output-dir", default="media_assets", help="Output directory for media assets")
    
    # Commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create script
    script_parser = subparsers.add_parser("script", help="Generate a script")
    script_parser.add_argument("--style", default="informative", 
                             choices=["informative", "entertaining", "educational", "controversial"],
                             help="Script style")
    
    # Download YouTube voice
    youtube_parser = subparsers.add_parser("youtube", help="Download and prepare YouTube voice")
    youtube_parser.add_argument("url", help="YouTube URL")
    youtube_parser.add_argument("transcript", help="Transcript of the voice sample")
    youtube_parser.add_argument("--id", help="Voice ID")
    youtube_parser.add_argument("--start", help="Start time (MM:SS)")
    youtube_parser.add_argument("--end", help="End time (MM:SS)")
    
    # Generate narration with voice clone
    voice_clone_parser = subparsers.add_parser("voice-clone", help="Generate narration using voice cloning")
    voice_clone_parser.add_argument("script", help="Path to script file")
    voice_clone_parser.add_argument("voice_sample", help="Path to voice sample WAV file")
    voice_clone_parser.add_argument("voice_transcript", help="Transcript text or path to transcript file")
    voice_clone_parser.add_argument("--output-dir", help="Directory to save output audio")
    voice_clone_parser.add_argument("--pitch", type=float, default=0.0, help="Pitch modification (-12 to 12)")
    voice_clone_parser.add_argument("--speed", type=float, default=1.0, help="Speed factor (0.5 to 2.0)")
    voice_clone_parser.add_argument("--brightness", type=float, default=0.0, help="Brightness (-1.0 to 1.0)")
    voice_clone_parser.add_argument("--expressiveness", type=float, default=0.0, help="Expressiveness (0.0 to 1.0)")
    voice_clone_parser.add_argument("--create-variants", action="store_true", help="Create voice variants")
    voice_clone_parser.add_argument("--preprocess", action="store_true", help="Preprocess voice sample")
    
    # Create complete short with voice clone
    short_clone_parser = subparsers.add_parser("short-clone", help="Create a complete YouTube Short with voice cloning")
    short_clone_parser.add_argument("--style", default="informative", 
                                  choices=["informative", "entertaining", "educational", "controversial"],
                                  help="Script style")
    short_clone_parser.add_argument("voice_sample", help="Path to voice sample WAV file")
    short_clone_parser.add_argument("voice_transcript", help="Transcript text or path to transcript file")
    short_clone_parser.add_argument("--pitch", type=float, default=0.0, help="Pitch modification (-12 to 12)")
    short_clone_parser.add_argument("--speed", type=float, default=1.0, help="Speed factor (0.5 to 2.0)")
    short_clone_parser.add_argument("--brightness", type=float, default=0.0, help="Brightness (-1.0 to 1.0)")
    short_clone_parser.add_argument("--expressiveness", type=float, default=0.0, help="Expressiveness (0.0 to 1.0)")
    short_clone_parser.add_argument("--create-variants", action="store_true", help="Create voice variants")
    short_clone_parser.add_argument("--no-preprocess", action="store_true", help="Skip preprocessing voice sample")
    
    args = parser.parse_args()
    
    # Initialize creator
    creator = EnhancedYouTubeShortsCreator(
        config_file=args.config,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Initialize components
    creator.initialize()
    
    # Execute command
    result = None
    
    if args.command == "script":
        result = creator.create_script(style=args.style)
        
    elif args.command == "youtube":
        result = creator.download_and_prepare_youtube_voice(
            args.url, args.transcript, args.id, args.start, args.end)
        
    elif args.command == "voice-clone":
        voice_settings = {
            "pitch": args.pitch,
            "speed": args.speed,
            "brightness": args.brightness,
            "expressiveness": args.expressiveness
        }
        result = creator.generate_narration_with_clone(
            args.script, 
            args.voice_sample, 
            args.voice_transcript,
            args.output_dir, 
            voice_settings, 
            args.create_variants, 
            args.preprocess
        )
    
    elif args.command == "short-clone":
        voice_settings = {
            "pitch": args.pitch,
            "speed": args.speed,
            "brightness": args.brightness,
            "expressiveness": args.expressiveness
        }
        result = creator.create_complete_short_with_clone(
            args.style, args.voice_sample, args.voice_transcript,
            voice_settings, args.create_variants, not args.no_preprocess
        )
    
    else:
        parser.print_help()
        return
    
    # Print result
    if result:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()