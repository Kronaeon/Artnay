"""
Integration example showing how the AudioGenerator fits into the YouTube Shorts pipeline.
This demonstrates the complete workflow from script generation to audio creation.
"""

import sys
import logging
from pathlib import Path
import argparse

# Import your pipeline components
sys.path.append(str(Path(__file__).parent))

from tubeShortsScriptGen import YouTubeShortsScriptGenerator
from audio_generator import AudioGenerator
from audio_utils import AudioUtils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

class YouTubeShortsContentCreator:
    """
    Main class that orchestrates the content creation pipeline.
    """
    
    def __init__(self, 
                 input_dir: str = "CleanSC",
                 output_dir: str = "media_assets",
                 model_configs: dict = None):
        """Initialize the content creator with specified directories."""
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create necessary directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)
        
        # Default model configurations
        self.model_configs = model_configs or {
            "script_model": {
                "model_path": "/home/horus/Projects/Models/gemma-3-finetune.Q8_0.gguf",
                "n_gpu_layers": 35,
                "n_ctx": 2048,
                "n_batch": 512
            },
            "audio_model": {
                "model_name": "openvoice_v2",
                "voice_sample": None
            }
        }
        
        # Initialize components
        self.script_generator = None
        self.audio_generator = None
    
    def initialize_script_generator(self):
        """Initialize the script generation component."""
        logging.info("Initializing script generator...")
        
        config = self.model_configs["script_model"]
        self.script_generator = YouTubeShortsScriptGenerator(
            input_dir=str(self.input_dir),
            output_dir="scripts",
            model_path=config["model_path"],
            n_gpu_layers=config["n_gpu_layers"],
            n_ctx=config["n_ctx"],
            n_batch=config["n_batch"]
        )
    
    def initialize_audio_generator(self, model_name: str = None):
        """Initialize the audio generation component."""
        logging.info("Initializing audio generator...")
        
        model_name = model_name or self.model_configs["audio_model"]["model_name"]
        self.audio_generator = AudioGenerator(
            output_dir=str(self.output_dir / "audio"),
            model_name=model_name,
            voice_sample=self.model_configs["audio_model"]["voice_sample"]
        )
    
    def create_content(self, style: str = "informative", voice_settings: dict = None):
        """
        Create complete content: script and audio narration.
        
        Args:
            style: Style of the content ("informative", "entertaining", etc.)
            voice_settings: Custom voice settings for audio generation
        """
        # Step 1: Generate script
        logging.info(f"Generating {style} script...")
        script_data = self.script_generator.generate_script(style=style)
        script_file = script_data["file_path"]
        
        # Step 2: Generate audio narration
        logging.info("Generating audio narration...")
        audio_files = self.audio_generator.generate_narration(
            script_file=script_file,
            voice_settings=voice_settings
        )
        
        return {
            "script": script_data,
            "audio_files": audio_files
        }
    
    def test_different_voices(self, script_file: str):
        """
        Test the same script with different TTS models.
        
        Args:
            script_file: Path to script file
        """
        models = AudioGenerator.list_available_models()
        results = {}
        
        for model_name in models:
            try:
                # Switch to new model
                self.audio_generator.load_model(model_name)
                
                # Generate audio
                audio_files = self.audio_generator.generate_narration(script_file)
                
                # Rename files to include model name
                renamed_files = []
                for audio_file in audio_files:
                    new_name = audio_file.parent / f"{model_name}_{audio_file.name}"
                    audio_file.rename(new_name)
                    renamed_files.append(new_name)
                
                results[model_name] = renamed_files
                logging.info(f"Successfully generated audio with {model_name}")
                
            except Exception as e:
                logging.error(f"Failed to generate audio with {model_name}: {e}")
                results[model_name] = None
        
        return results
    
    def create_voice_profile(self, voice_sample: str):
        """
        Create a voice profile and set up voice cloning.
        
        Args:
            voice_sample: Path to voice sample file
        """
        # Create voice profile
        profile = AudioUtils.create_voice_profile(voice_sample)
        
        # Set up voice cloning
        success = self.audio_generator.clone_voice(voice_sample)
        
        return {
            "profile": profile,
            "cloning_successful": success
        }
    
    def generate_complete_content_package(self, 
                                        styles: list = None, 
                                        voice_sample: str = None):
        """
        Generate a complete content package with multiple variations.
        
        Args:
            styles: List of styles to generate
            voice_sample: Optional voice sample for cloning
        """
        styles = styles or ["informative", "entertaining", "educational"]
        results = []
        
        # Set up voice cloning if sample provided
        if voice_sample:
            voice_profile = self.create_voice_profile(voice_sample)
            logging.info("Voice profile created and cloning set up")
        
        # Generate content for each style
        for style in styles:
            try:
                content = self.create_content(style=style)
                results.append({
                    "style": style,
                    "status": "success",
                    "data": content
                })
                logging.info(f"Successfully created {style} content")
                
            except Exception as e:
                results.append({
                    "style": style,
                    "status": "failed",
                    "error": str(e)
                })
                logging.error(f"Failed to create {style} content: {e}")
        
        return results
    
    def benchmark_system(self):
        """
        Run system benchmarks to evaluate performance.
        """
        logging.info("Running system benchmarks...")
        
        # Create test script
        test_script = AudioUtils.create_test_script()
        
        # Benchmark all audio models
        audio_results = AudioUtils.test_all_models()
        
        # Generate comparison report
        AudioUtils.generate_model_comparison_report(audio_results)
        
        return {
            "audio_benchmark": audio_results,
            "test_script": test_script
        }


def main():
    """Main function to demonstrate the pipeline usage."""
    parser = argparse.ArgumentParser(description="YouTube Shorts Content Creation Pipeline")
    parser.add_argument("--input-dir", default="CleanSC", help="Input directory with cleaned content")
    parser.add_argument("--output-dir", default="media_assets", help="Output directory for media assets")
    parser.add_argument("--style", default="informative", help="Content style")
    parser.add_argument("--audio-model", default="openvoice_v2", help="TTS model to use")
    parser.add_argument("--voice-sample", help="Path to voice sample for cloning")
    parser.add_argument("--test-all", action="store_true", help="Test all audio models")
    parser.add_argument("--benchmark", action="store_true", help="Run system benchmarks")
    
    args = parser.parse_args()
    
    # Initialize content creator
    creator = YouTubeShortsContentCreator(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Initialize components
    creator.initialize_script_generator()
    creator.initialize_audio_generator(model_name=args.audio_model)
    
    try:
        if args.benchmark:
            # Run benchmarks
            results = creator.benchmark_system()
            print("\nBenchmark Results:")
            print(f"See 'model_comparison.json' for detailed results")
            
        elif args.test_all:
            # Generate script first
            script_data = creator.script_generator.generate_script(style=args.style)
            script_file = script_data["file_path"]
            
            # Test all audio models
            voice_results = creator.test_different_voices(script_file)
            print("\nVoice Test Results:")
            for model, files in voice_results.items():
                if files:
                    print(f"{model}: Generated {len(files)} audio files")
                else:
                    print(f"{model}: Failed to generate audio")
        
        else:
            # Generate complete content package
            styles = ["informative", "entertaining", "educational"]
            results = creator.generate_complete_content_package(
                styles=styles,
                voice_sample=args.voice_sample
            )
            
            print("\nContent Generation Summary:")
            for result in results:
                print(f"{result['style']}: {result['status']}")
                if result['status'] == 'success':
                    script_title = result['data']['script']['title']
                    audio_count = len(result['data']['audio_files'])
                    print(f"  Script: {script_title}")
                    print(f"  Audio files: {audio_count}")
    
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()