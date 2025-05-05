import argparse
import logging
import torch
import soundfile as sf
import numpy as np
import os
from pathlib import Path
from dia.model import Dia

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


import os
import re
import logging
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import subprocess

# Remove or simplify the previous blending functions and add these simpler ones:

def simple_synchronize(audio1, audio2, sr):
    """
    Simple time-stretching approach that avoids complex library dependencies
    """
    # Get durations
    dur1 = len(audio1) / sr
    dur2 = len(audio2) / sr
    
    # Calculate time stretch factor
    stretch_factor = dur1 / dur2
    
    # Only stretch if there's a significant difference
    if abs(1 - stretch_factor) > 0.05:
        try:
            # Use scipy's resample function which has fewer dependencies
            from scipy import signal
            logging.info(f"Resampling voice 2 with factor {stretch_factor:.3f}")
            
            # Calculate new length
            new_len = int(len(audio2) / stretch_factor)
            
            # Resample
            audio2_synced = signal.resample(audio2, new_len)
            return audio1, audio2_synced
        except Exception as e:
            logging.warning(f"Resampling failed: {e}, using original audio")
    
    return audio1, audio2

def simple_blend(audio1, audio2, ratio=0.5):
    """
    Simple, robust blending approach
    """
    # Ensure same length
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    # Apply cosine crossfade windows
    fade_len = min_len
    fade_in = np.cos(np.linspace(np.pi, 0, fade_len)) * 0.5 + 0.5
    fade_out = np.cos(np.linspace(0, np.pi, fade_len)) * 0.5 + 0.5
    
    # Apply weighted fade
    audio1_fade = audio1 * fade_out * (1 - ratio)
    audio2_fade = audio2 * fade_in * ratio
    
    # Mix
    blended = audio1_fade + audio2_fade
    
    # Normalize
    if np.max(np.abs(blended)) > 0:
        blended = blended / np.max(np.abs(blended)) * 0.9
    
    return blended

def clean_audio_simple(audio):
    """
    Very simple audio cleaning without complex dependencies
    """
    # Simple normalization
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    
    return audio


# ======== New addtional blending methods end ========


def preprocess_audio(audio_file, output_dir="preprocessed"):
    """
    Preprocess audio to ensure compatibility with Dia model
    """
    try:
        from pydub import AudioSegment
        import librosa
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the audio file
        audio = AudioSegment.from_file(audio_file)
        
        # Convert to WAV if not already
        if not audio_file.lower().endswith('.wav'):
            audio_file_wav = os.path.join(output_dir, os.path.basename(audio_file).split('.')[0] + '.wav')
            audio.export(audio_file_wav, format="wav")
        else:
            audio_file_wav = audio_file
            
        # Ensure mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
            audio.export(audio_file_wav, format="wav")
            
        # Resample to 44.1kHz if needed
        y, sr = librosa.load(audio_file_wav, sr=None)
        if sr != 44100:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=44100)
            sf.write(audio_file_wav, y_resampled, 44100)
            
        # Normalize audio levels
        y, sr = librosa.load(audio_file_wav, sr=None)
        y_norm = librosa.util.normalize(y)
        sf.write(audio_file_wav, y_norm, sr)
        
        logging.info(f"Preprocessed audio saved to {audio_file_wav}")
        return audio_file_wav
        
    except Exception as e:
        logging.error(f"Error preprocessing audio: {e}")
        return audio_file

def main():
    parser = argparse.ArgumentParser(description="Enhanced Voice Blending with Dia")
    parser.add_argument("voice1_wav", help="Path to first voice WAV file")
    parser.add_argument("voice1_txt", help="Path to first voice transcript TXT file")
    parser.add_argument("voice2_wav", help="Path to second voice WAV file") 
    parser.add_argument("voice2_txt", help="Path to second voice transcript TXT file")
    parser.add_argument("--ratio", type=float, default=0.5, help="Blend ratio (0.0-1.0)")
    parser.add_argument("--output-dir", default="media_assets/audio", help="Output directory")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess audio files")
    parser.add_argument("--test-text", default="Well, here we go again. Quickly now—read this line with purpose, pause... then shift tone. Do rising questions sound right? Or do they fall flat? Either way: crisp diction, varied rhythm, full voice range. That should do it, wouldn’t you say?", 
                        help="Text to synthesize with blended voice")
    parser.add_argument("--model-path", default="/home/horus/Projects/Models/AUDIOMODELS/nari-labs",
                       help="Path to Dia model")
    parser.add_argument("--prompt-length", type=int, default=3, 
                       help="Maximum seconds of reference audio to use")
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Preprocess audio if requested
        if args.preprocess:
            logging.info("Preprocessing audio files...")
            preprocess_dir = os.path.join(args.output_dir, "preprocessed")
            voice1_wav = preprocess_audio(args.voice1_wav, preprocess_dir)
            voice2_wav = preprocess_audio(args.voice2_wav, preprocess_dir)
        else:
            voice1_wav = args.voice1_wav
            voice2_wav = args.voice2_wav
            
        # Load transcripts
        with open(args.voice1_txt, 'r') as f:
            voice1_transcript = f.read().strip()
        
        with open(args.voice2_txt, 'r') as f:
            voice2_transcript = f.read().strip()
        
        # Format transcripts with speaker tags if they don't already have them
        if not voice1_transcript.startswith("[S"):
            voice1_transcript = f"[S1] {voice1_transcript}"
            
        if not voice2_transcript.startswith("[S"):
            voice2_transcript = f"[S1] {voice2_transcript}"
            
        # Format test text
        test_text = f"[S1] {args.test_text}"
        
        # Load model
        logging.info(f"Loading Dia model from {args.model_path}")
        model = Dia.from_pretrained(args.model_path, compute_dtype="float16")
        
        # INDIVIDUAL VOICE GENERATION FIRST
        # Generate audio with first voice
        logging.info("Generating audio with voice 1...")
        voice1_output_path = os.path.join(args.output_dir, "voice1_output.wav")
        
        try:
            # Generate using the approach from voice_clone.py example
            audio1 = model.generate(
                voice1_transcript + test_text, 
                audio_prompt=voice1_wav,
                use_torch_compile=True,
                verbose=True
            )
            
            # Save audio using the model's built-in method
            model.save_audio(voice1_output_path, audio1)
            logging.info(f"Voice 1 audio saved to {voice1_output_path}")
            
            # Separately save a sample with just the voice1 transcript for reference
            voice1_ref_path = os.path.join(args.output_dir, "voice1_reference.wav")
            ref_audio1 = model.generate(
                voice1_transcript, 
                audio_prompt=voice1_wav,
                use_torch_compile=True,
                verbose=True
            )
            model.save_audio(voice1_ref_path, ref_audio1)
            
        except Exception as e:
            logging.error(f"Error generating voice 1 audio: {e}")
            return
        
        # Generate audio with second voice
        logging.info("Generating audio with voice 2...")
        voice2_output_path = os.path.join(args.output_dir, "voice2_output.wav")
        
        try:
            # Try a slightly different approach for voice 2
            audio2 = model.generate(
                voice2_transcript + test_text,
                audio_prompt=voice2_wav,
                use_torch_compile=True,
                verbose=True
            )
            
            model.save_audio(voice2_output_path, audio2)
            logging.info(f"Voice 2 audio saved to {voice2_output_path}")
            
            # Separately save a sample with just the voice2 transcript for reference
            voice2_ref_path = os.path.join(args.output_dir, "voice2_reference.wav")
            ref_audio2 = model.generate(
                voice2_transcript, 
                audio_prompt=voice2_wav,
                use_torch_compile=True,
                verbose=True
            )
            model.save_audio(voice2_ref_path, ref_audio2)
            
        except Exception as e:
            logging.error(f"Error generating voice 2 audio: {e}")
            # Continue with blending attempt using shorter segments
            
        # BLENDING APPROACH 1: Direct generate with blended prompt (if both voices worked)
        try:
            if 'audio1' in locals() and 'audio2' in locals():
                logging.info("Attempting voice blending approach 1: Direct generate with blended prompt")
                
                # Try generating with a combination of both voice transcripts
                blended_text = f"[S1] {voice1_transcript} {voice2_transcript} {args.test_text}"
                
                # Alternating prompts approach
                if args.ratio <= 0.5:
                    primary_prompt = voice1_wav
                    ratio_text = f"Make it sound {int(args.ratio*100)}% like the second voice."
                else:
                    primary_prompt = voice2_wav
                    ratio_text = f"Make it sound {int((1-args.ratio)*100)}% like the first voice."
                
                # Add ratio instruction to the text prompt
                blended_text = f"[S1] {ratio_text} {args.test_text}"
                
                blended_audio1 = model.generate(
                    blended_text,
                    audio_prompt=primary_prompt,
                    use_torch_compile=True,
                    verbose=True
                )
                
                blended_output_path1 = os.path.join(args.output_dir, "blended_voice_approach1.wav")
                model.save_audio(blended_output_path1, blended_audio1)
                logging.info(f"Blended voice (approach 1) saved to {blended_output_path1}")
            
        except Exception as e:
            logging.error(f"Error in blending approach 1: {e}")
        

        # SIMPLIFIED ROBUST BLENDING APPROACH
        try:
            logging.info("Attempting simplified robust voice blending")
            
            # Load the outputs as raw audio data
            try:
                raw_audio1, sr1 = sf.read(voice1_output_path)
            except:
                logging.warning("Could not read voice1_output.wav, trying to use the model's raw output")
                raw_audio1 = audio1  # Use the raw output from model
                sr1 = 44100  # Assuming 44.1kHz
                
            try:
                raw_audio2, sr2 = sf.read(voice2_output_path)
            except:
                logging.warning("Could not read voice2_output.wav, trying to use the model's raw output")
                if 'audio2' in locals():
                    raw_audio2 = audio2  # Use the raw output from model
                    sr2 = 44100  # Assuming 44.1kHz
                else:
                    logging.error("No voice2 audio available for blending")
                    raw_audio2 = None
            
            if raw_audio1 is not None and raw_audio2 is not None:
                # Ensure both have the same sample rate
                if sr1 != sr2:
                    try:
                        from scipy import signal
                        raw_audio2 = signal.resample(raw_audio2, int(len(raw_audio2) * sr1 / sr2))
                        sr2 = sr1
                    except Exception as e:
                        logging.error(f"Error resampling: {e}")
                        # Continue with potentially mismatched sample rates
                
                # Step 1: Synchronize with simple approach
                logging.info("Step 1: Basic voice synchronization")
                sync_audio1, sync_audio2 = simple_synchronize(raw_audio1, raw_audio2, sr1)
                
                # Save synchronized audio
                sync1_path = os.path.join(args.output_dir, "voice1_synced.wav")
                sync2_path = os.path.join(args.output_dir, "voice2_synced.wav")
                sf.write(sync1_path, sync_audio1, sr1)
                sf.write(sync2_path, sync_audio2, sr1)
                logging.info(f"Synchronized voices saved")
                
                # Step 2: Create a few different blends with different ratios
                logging.info("Step 2: Creating blended variants")
                
                # Standard blend
                blend_standard = simple_blend(sync_audio1, sync_audio2, ratio=args.ratio)
                standard_path = os.path.join(args.output_dir, "blended_standard.wav")
                sf.write(standard_path, blend_standard, sr1)
                
                # Voice 1 dominant (25% voice 2)
                blend_v1 = simple_blend(sync_audio1, sync_audio2, ratio=0.25)
                v1_path = os.path.join(args.output_dir, "blended_v1_dominant.wav")
                sf.write(v1_path, blend_v1, sr1)
                
                # Voice 2 dominant (75% voice 2)
                blend_v2 = simple_blend(sync_audio1, sync_audio2, ratio=0.75)
                v2_path = os.path.join(args.output_dir, "blended_v2_dominant.wav")
                sf.write(v2_path, blend_v2, sr1)
                
                # Clean the standard blend
                final_audio = clean_audio_simple(blend_standard)
                
                # Save final output
                final_output_path = os.path.join(args.output_dir, "blended_voice.wav")
                sf.write(final_output_path, final_audio, sr1)
                logging.info(f"Final blended voice saved to {final_output_path}")
                
                # Also save approach 1 as a possible fallback
                try:
                    approach1_path = os.path.join(args.output_dir, "blended_voice_approach1.wav")
                    if os.path.exists(approach1_path):
                        fallback_path = os.path.join(args.output_dir, "blended_voice_fallback.wav")
                        subprocess.run(['cp', approach1_path, fallback_path], check=True)
                        logging.info(f"Approach 1 saved as fallback at {fallback_path}")
                except Exception as e:
                    logging.error(f"Error saving fallback: {e}")

        except Exception as e:
            logging.error(f"Error in simplified blending approach: {e}")
            
            # Ultimate fallback - just use voice1_output.wav
            try:
                if os.path.exists(voice1_output_path):
                    final_output_path = os.path.join(args.output_dir, "blended_voice.wav")
                    subprocess.run(['cp', voice1_output_path, final_output_path], check=True)
                    logging.info(f"Using voice1 as ultimate fallback at {final_output_path}")
            except Exception as e2:
                logging.error(f"Error in ultimate fallback: {e2}")
            
    except Exception as e:
        logging.error(f"Error in voice blending process: {e}")

if __name__ == "__main__":
    main()