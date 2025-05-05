import os
import re
import logging
import argparse
import json
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VoicePreparation:
    """
    Utilities for preparing and managing voice samples for Dia voice cloning.
    """
    
    def __init__(self, output_dir="voice_samples"):
        """Initialize the voice preparation utilities."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voice_profiles = {}
    
    def download_youtube_audio(self, youtube_url: str, output_name: str, 
                              start_time: Optional[str] = None,
                              end_time: Optional[str] = None) -> Optional[Path]:
        """
        Download audio from a YouTube video.
        
        Args:
            youtube_url: URL of the YouTube video
            output_name: Name for the output file
            start_time: Start time in format 'MM:SS' (optional)
            end_time: End time in format 'MM:SS' (optional)
            
        Returns:
            Path to downloaded audio file or None if failed
        """
        try:
            # Validate URL
            if not re.match(r'https?://(www\.)?(youtube\.com|youtu\.be)', youtube_url):
                logging.error(f"Invalid YouTube URL: {youtube_url}")
                return None
            
            output_file = self.output_dir / f"{output_name}.mp3"
            
            # Check for yt-dlp first, then youtube-dl
            download_cmd = "yt-dlp"
            
            try:
                subprocess.run([download_cmd, "--version"], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                download_cmd = "youtube-dl"
                try:
                    subprocess.run([download_cmd, "--version"], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                except (subprocess.SubprocessError, FileNotFoundError):
                    logging.error("Neither yt-dlp nor youtube-dl found. Please install one.")
                    return None
            
            # Base command
            cmd = [
                download_cmd,
                "-x",  # Extract audio
                "--audio-format", "mp3",
                "--audio-quality", "0",  # Best quality
                "-o", f"{output_file.with_suffix('.%(ext)s')}",
                youtube_url
            ]
            
            logging.info(f"Downloading audio from {youtube_url}")
            subprocess.run(cmd, check=True)
            
            # Trim audio if needed
            if start_time or end_time:
                return self.trim_audio(output_file, start_time, end_time)
            
            logging.info(f"Audio downloaded to {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"Error downloading audio: {e}")
            return None
    
    def trim_audio(self, 
                  audio_file: Path, 
                  start_time: Optional[str] = None, 
                  end_time: Optional[str] = None) -> Path:
        """
        Trim audio file to specified start and end times.
        
        Args:
            audio_file: Path to audio file
            start_time: Start time in format 'MM:SS'
            end_time: End time in format 'MM:SS'
            
        Returns:
            Path to trimmed audio file
        """
        try:
            # Convert input file to Path if it's a string
            audio_file = Path(audio_file)
            
            # If no trimming needed, return original file
            if not start_time and not end_time:
                return audio_file
            
            # Parse start and end times
            start_ms = 0
            if start_time:
                parts = start_time.split(':')
                if len(parts) == 2:
                    start_ms = (int(parts[0]) * 60 + int(parts[1])) * 1000
                elif len(parts) == 3:
                    start_ms = (int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])) * 1000
            
            # Load audio file
            audio = AudioSegment.from_file(str(audio_file))
            
            end_ms = len(audio)
            if end_time:
                parts = end_time.split(':')
                if len(parts) == 2:
                    end_ms = (int(parts[0]) * 60 + int(parts[1])) * 1000
                elif len(parts) == 3:
                    end_ms = (int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])) * 1000
            
            # Trim audio
            trimmed_audio = audio[start_ms:end_ms]
            
            # Create output filename
            output_file = audio_file.with_name(f"{audio_file.stem}_trimmed{audio_file.suffix}")
            
            # Export trimmed audio
            trimmed_audio.export(output_file, format=output_file.suffix.lstrip('.'))
            
            logging.info(f"Trimmed audio saved to {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"Error trimming audio: {e}")
            return audio_file
    
    def normalize_audio(self, audio_file: Path, target_db: float = -20.0) -> Path:
        """
        Normalize audio to target loudness.
        
        Args:
            audio_file: Path to audio file
            target_db: Target peak dB level
            
        Returns:
            Path to normalized audio file
        """
        try:
            audio_file = Path(audio_file)
            
            # Load audio
            y, sr = librosa.load(str(audio_file))
            
            # Calculate current peak dB
            current_peak_db = 20 * np.log10(np.max(np.abs(y)))
            
            # Calculate scaling factor
            scaling_factor = 10 ** ((target_db - current_peak_db) / 20)
            
            # Apply normalization
            y_normalized = y * scaling_factor
            
            # Create output filename
            output_file = audio_file.with_name(f"{audio_file.stem}_normalized{audio_file.suffix}")
            
            # Save normalized audio
            sf.write(str(output_file), y_normalized, sr)
            
            logging.info(f"Normalized audio saved to {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"Error normalizing audio: {e}")
            return audio_file
    
    def analyze_voice_sample(self, audio_file: Path) -> Dict[str, Any]:
        """
        Analyze voice sample for quality and characteristics.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary with voice analysis
        """
        try:
            audio_file = Path(audio_file)
            
            # Load audio
            y, sr = librosa.load(str(audio_file))
            
            # Extract voice features
            duration = len(y) / sr
            
            # Pitch estimation
            pitch = librosa.yin(y, fmin=50, fmax=500, sr=sr)
            valid_pitch = pitch[pitch > 0]
            
            analysis = {
                "filename": audio_file.name,
                "sample_rate": sr,
                "duration": duration,
                "format": audio_file.suffix.lstrip('.'),
                "quality": {
                    "rms_energy": float(np.sqrt(np.mean(y**2))),
                    "peak_amplitude": float(np.max(np.abs(y))),
                    "silent_percentage": float(np.mean(np.abs(y) < 0.01) * 100)
                },
                "voice_characteristics": {
                    "mean_pitch": float(np.mean(valid_pitch)) if len(valid_pitch) > 0 else 0,
                    "pitch_std": float(np.std(valid_pitch)) if len(valid_pitch) > 0 else 0,
                    "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
                }
            }
            
            # Quality assessment
            quality_score = 0
            quality_issues = []
            
            # Check duration
            if duration < 3:
                quality_issues.append("Sample too short (< 3 seconds)")
            elif duration > 20:
                quality_issues.append("Sample too long (> 20 seconds)")
            else:
                quality_score += 25  # Good duration
            
            # Check volume level
            if analysis["quality"]["rms_energy"] < 0.01:
                quality_issues.append("Audio level too low")
            elif analysis["quality"]["peak_amplitude"] > 0.99:
                quality_issues.append("Audio may be clipping")
            else:
                quality_score += 25  # Good levels
            
            # Check silence
            if analysis["quality"]["silent_percentage"] > 30:
                quality_issues.append(f"Contains too much silence ({analysis['quality']['silent_percentage']:.1f}%)")
            else:
                quality_score += 25  # Not too silent
            
            # Voice clarity (using spectral centroid as proxy)
            if 500 < analysis["voice_characteristics"]["spectral_centroid"] < 2000:
                quality_score += 25  # Good voice frequency range
            
            analysis["quality_score"] = quality_score
            analysis["quality_issues"] = quality_issues
            
            # Save voice profile
            self.voice_profiles[audio_file.stem] = analysis
            
            logging.info(f"Voice analysis completed for {audio_file.name} (score: {quality_score}/100)")
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing voice: {e}")
            return {"error": str(e)}
    
    def prepare_training_sample(self, 
                               audio_file: Path, 
                               transcript: str,
                               voice_id: str = None) -> Dict[str, Any]:
        """
        Prepare a voice sample for training.
        
        Args:
            audio_file: Path to audio file
            transcript: Transcript of the audio
            voice_id: ID for this voice
            
        Returns:
            Dictionary with prepared sample info
        """
        try:
            audio_file = Path(audio_file)
            
            if not audio_file.exists():
                logging.error(f"Audio file not found: {audio_file}")
                return {"success": False, "error": "File not found"}
            
            voice_id = voice_id or audio_file.stem
            
            # Analyze sample
            analysis = self.analyze_voice_sample(audio_file)
            
            # Create metadata
            metadata = {
                "voice_id": voice_id,
                "audio_file": str(audio_file),
                "transcript": transcript,
                "analysis": analysis
            }
            
            # Save metadata
            metadata_file = self.output_dir / f"{voice_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Prepare formatted transcript with speaker tags
            formatted_transcript = f"[S1] {transcript}"
            
            # Return results
            return {
                "success": True,
                "voice_id": voice_id,
                "audio_file": str(audio_file),
                "transcript": transcript,
                "formatted_transcript": formatted_transcript,
                "metadata_file": str(metadata_file)
            }
            
        except Exception as e:
            logging.error(f"Error preparing training sample: {e}")
            return {"success": False, "error": str(e)}
    
    def extract_snippets(self, 
                        audio_file: Path,
                        snippets_dir: Optional[Path] = None,
                        min_duration: float = 5.0,
                        max_duration: float = 15.0) -> List[Path]:
        """
        Extract voice snippets from longer audio.
        
        Args:
            audio_file: Path to audio file
            snippets_dir: Directory to save snippets (default: output_dir/snippets)
            min_duration: Minimum snippet duration in seconds
            max_duration: Maximum snippet duration in seconds
            
        Returns:
            List of paths to extracted snippets
        """
        try:
            audio_file = Path(audio_file)
            
            # Set up snippets directory
            if snippets_dir is None:
                snippets_dir = self.output_dir / "snippets" / audio_file.stem
            else:
                snippets_dir = Path(snippets_dir)
            
            snippets_dir.mkdir(parents=True, exist_ok=True)
            
            # Load audio
            y, sr = librosa.load(str(audio_file))
            
            # Detect voice activity
            import librosa.effects as effects
            
            # Split on silence
            intervals = librosa.effects.split(y, top_db=30)
            
            snippets = []
            
            for i, (start, end) in enumerate(intervals):
                # Convert samples to seconds
                start_sec = start / sr
                end_sec = end / sr
                duration = end_sec - start_sec
                
                # Filter by duration
                if min_duration <= duration <= max_duration:
                    # Extract snippet
                    snippet = y[start:end]
                    
                    # Save snippet
                    snippet_file = snippets_dir / f"snippet_{i:03d}_{start_sec:.1f}_{end_sec:.1f}.wav"
                    sf.write(str(snippet_file), snippet, sr)
                    
                    snippets.append(snippet_file)
                    logging.info(f"Saved snippet {i}: {duration:.2f}s to {snippet_file}")
            
            logging.info(f"Extracted {len(snippets)} snippets from {audio_file}")
            return snippets
            
        except Exception as e:
            logging.error(f"Error extracting snippets: {e}")
            return []


def main():
    """Command line interface for voice preparation tools."""
    parser = argparse.ArgumentParser(description="Voice Sample Preparation Tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download audio from YouTube")
    download_parser.add_argument("url", help="YouTube URL")
    download_parser.add_argument("output_name", help="Output filename")
    download_parser.add_argument("--start", help="Start time (MM:SS)")
    download_parser.add_argument("--end", help="End time (MM:SS)")
    
    # Trim command
    trim_parser = subparsers.add_parser("trim", help="Trim audio file")
    trim_parser.add_argument("audio_file", help="Audio file to trim")
    trim_parser.add_argument("--start", help="Start time (MM:SS)")
    trim_parser.add_argument("--end", help="End time (MM:SS)")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze voice sample")
    analyze_parser.add_argument("audio_file", help="Audio file to analyze")
    
    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare voice sample for training")
    prepare_parser.add_argument("audio_file", help="Audio file")
    prepare_parser.add_argument("transcript", help="Transcript of the audio")
    prepare_parser.add_argument("--voice-id", help="Voice ID")
    
    # Extract snippets command
    extract_parser = subparsers.add_parser("extract", help="Extract voice snippets")
    extract_parser.add_argument("audio_file", help="Audio file")
    extract_parser.add_argument("--min-duration", type=float, default=5.0, help="Minimum snippet duration")
    extract_parser.add_argument("--max-duration", type=float, default=15.0, help="Maximum snippet duration")
    
    args = parser.parse_args()
    
    # Initialize voice preparation
    voice_prep = VoicePreparation()
    
    if args.command == "download":
        output_file = voice_prep.download_youtube_audio(
            args.url, args.output_name, args.start, args.end)
        if output_file:
            print(f"Downloaded audio to {output_file}")
    
    elif args.command == "trim":
        output_file = voice_prep.trim_audio(args.audio_file, args.start, args.end)
        if output_file:
            print(f"Trimmed audio saved to {output_file}")
    
    elif args.command == "analyze":
        analysis = voice_prep.analyze_voice_sample(args.audio_file)
        print(json.dumps(analysis, indent=2))
    
    elif args.command == "prepare":
        result = voice_prep.prepare_training_sample(
            args.audio_file, args.transcript, args.voice_id)
        print(json.dumps(result, indent=2))
    
    elif args.command == "extract":
        snippets = voice_prep.extract_snippets(
            args.audio_file, min_duration=args.min_duration, max_duration=args.max_duration)
        print(f"Extracted {len(snippets)} snippets")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
