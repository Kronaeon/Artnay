import argparse
import logging
import torch
import soundfile as sf
import numpy as np
import os
import librosa
import librosa.effects
from pathlib import Path
from dia.model import Dia

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Voice modification functions
def modify_pitch(audio, sr, semitones=0):
    """
    Modify the pitch of audio by the specified number of semitones.
    """
    if semitones == 0:
        return audio
    
    try:
        # Use librosa for pitch shifting
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
    except Exception as e:
        logging.error(f"Error in pitch shifting: {e}")
        return audio

def modify_speed(audio, sr, speed_factor=1.0):
    """
    Modify the speed of audio without changing pitch.
    """
    if speed_factor == 1.0:
        return audio
    
    try:
        # Use librosa for time stretching
        return librosa.effects.time_stretch(audio, rate=speed_factor)
    except Exception as e:
        logging.error(f"Error in time stretching: {e}")
        return audio

def modify_timbre(audio, sr, brightness=0.0):
    """
    Modify the timbre/brightness of the voice.
    """
    if brightness == 0.0:
        return audio
    
    try:
        # Simple EQ-based approach for timbre modification
        # Convert to frequency domain
        D = librosa.stft(audio)
        
        # Create a filter that emphasizes/de-emphasizes higher frequencies
        freq_bins = D.shape[0]
        filter_curve = np.linspace(1.0 - brightness/2, 1.0 + brightness, freq_bins)
        
        # Apply filter
        D_filtered = D * filter_curve[:, np.newaxis]
        
        # Convert back to time domain
        return librosa.istft(D_filtered)
    except Exception as e:
        logging.error(f"Error in timbre modification: {e}")
        return audio

def modify_expressiveness(audio, sr, expressiveness=0.0):
    """
    Modify the expressiveness of the voice by enhancing dynamic range.
    """
    if expressiveness == 0.0:
        return audio
    
    try:
        # Simplified approach - dynamic range control
        # Extract envelope
        envelope = np.abs(audio)
        smoothed_envelope = np.convolve(envelope, np.ones(2048)/2048, mode='same')
        normalized_envelope = smoothed_envelope / np.max(smoothed_envelope) if np.max(smoothed_envelope) > 0 else smoothed_envelope
        
        # Apply dynamic modification
        centered = audio - np.mean(audio)
        factor = 1.0 + expressiveness * (normalized_envelope - np.mean(normalized_envelope))
        modified = centered * factor
        
        # Normalize
        return librosa.util.normalize(modified) * 0.9
    except Exception as e:
        logging.error(f"Error in expressiveness modification: {e}")
        return audio

def save_audio_file(audio, sr, output_path):
    """
    Save audio to file with proper checks.
    """
    try:
        # Ensure audio is normalized
        if np.max(np.abs(audio)) > 0.0:
            audio = librosa.util.normalize(audio) * 0.9
        
        # Save audio
        sf.write(output_path, audio, sr)
        logging.info(f"Audio saved to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving audio: {e}")
        return False

def preprocess_audio(audio_file, output_dir="preprocessed"):
    """
    Preprocess audio to ensure compatibility with Dia model
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio file
        y, sr = librosa.load(str(audio_file), sr=None)
        
        # Ensure mono (if stereo)
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = librosa.to_mono(y)
        
        # Resample to 44.1kHz if needed
        if sr != 44100:
            y = librosa.resample(y, orig_sr=sr, target_sr=44100)
            sr = 44100
            
        # Normalize audio levels
        y = librosa.util.normalize(y) * 0.9
        
        # Save preprocessed audio
        audio_file_wav = os.path.join(output_dir, os.path.basename(str(audio_file)).split('.')[0] + '.wav')
        sf.write(audio_file_wav, y, sr)
        
        logging.info(f"Preprocessed audio saved to {audio_file_wav}")
        return audio_file_wav
        
    except Exception as e:
        logging.error(f"Error preprocessing audio: {e}")
        return str(audio_file)

def extract_test_audio(audio_file):
    """
    Analyze and extract the test portion of generated audio.
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=None)
        
        # Get audio duration in seconds
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Try to detect significant silence that might separate reference from test
        # First normalize to ensure consistent silence detection
        y_norm = librosa.util.normalize(y)
        
        # Detect non-silent intervals
        intervals = librosa.effects.split(y_norm, top_db=30)
        
        if len(intervals) > 1:
            # Find the longest gap between intervals
            gaps = [intervals[i+1][0] - intervals[i][1] for i in range(len(intervals)-1)]
            if gaps:
                largest_gap_idx = np.argmax(gaps)
                
                # Use the point after the largest gap as the start of test audio
                test_start_idx = intervals[largest_gap_idx+1][0]
                
                # Check if this is a reasonable splitting point (not too early, not too late)
                split_ratio = test_start_idx / len(y)
                
                if 0.3 < split_ratio < 0.7:
                    logging.info(f"Extracted test audio starting at {split_ratio:.2f} of total duration")
                    return y[test_start_idx:], sr
        
        # Fallback: use the second half of the audio
        half_point = len(y) // 2
        logging.info("Using second half of audio as test portion")
        return y[half_point:], sr
    
    except Exception as e:
        logging.error(f"Error extracting test audio: {e}")
        # Return the original audio
        try:
            y, sr = librosa.load(audio_file, sr=None)
            return y, sr
        except:
            return None, None

def main():
    parser = argparse.ArgumentParser(description="Voice Cloning with Modifications")
    parser.add_argument("voice_wav", help="Path to voice WAV file")
    parser.add_argument("voice_txt", help="Path to voice transcript TXT file")
    parser.add_argument("--output-dir", default="media_assets/audio", help="Output directory")
    parser.add_argument("--pitch", type=float, default=0.0, help="Pitch modification in semitones (-12 to 12)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor (0.5 to 2.0)")
    parser.add_argument("--brightness", type=float, default=0.0, help="Brightness/timbre (-1.0 to 1.0)")
    parser.add_argument("--expressiveness", type=float, default=0.0, help="Expressiveness (0.0 to 1.0)")
    parser.add_argument("--test-text", default="Well, here we go again. Quickly now—read this line with purpose, pause... then shift tone. Do rising questions sound right? Or do they fall flat? Either way: crisp diction, varied rhythm, full voice range. That should do it, wouldn't you say?", 
                        help="Text to synthesize with cloned voice")
    parser.add_argument("--model-path", default="/home/horus/Projects/Models/AUDIOMODELS/nari-labs",
                       help="Path to Dia model")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess audio file")
    parser.add_argument("--create-variants", action="store_true", help="Create multiple variants with different settings")
    parser.add_argument("--skip-reference", action="store_true", help="Skip generating reference audio (use existing file)")
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Preprocess audio if requested
        if args.preprocess:
            logging.info("Preprocessing audio file...")
            preprocess_dir = os.path.join(args.output_dir, "preprocessed")
            voice_wav = preprocess_audio(args.voice_wav, preprocess_dir)
        else:
            voice_wav = args.voice_wav
            
        # Load transcript
        with open(args.voice_txt, 'r') as f:
            voice_transcript = f.read().strip()
        
        # Format transcript with speaker tags if it doesn't already have them
        if not voice_transcript.startswith("[S"):
            voice_transcript = f"[S1] {voice_transcript}"
            
        # Format test text
        test_text = f"[S1] {args.test_text}"
        
        # Define output paths
        reference_output_path = os.path.join(args.output_dir, "cloned_reference.wav")
        
        # Generate reference audio if needed
        if not args.skip_reference or not os.path.exists(reference_output_path):
            # Load model
            logging.info(f"Loading Dia model from {args.model_path}")
            model = Dia.from_pretrained(args.model_path, compute_dtype="float16")
            
            logging.info("Generating reference audio with cloned voice...")
            try:
                # Generate using the approach from voice_clone.py example
                audio = model.generate(
                    voice_transcript + test_text, 
                    audio_prompt=voice_wav,
                    use_torch_compile=True,
                    verbose=True
                )
                
                # Save audio using the model's built-in method
                model.save_audio(reference_output_path, audio)
                logging.info(f"Reference cloned audio saved to {reference_output_path}")
                
            except Exception as e:
                logging.error(f"Error generating reference audio: {e}")
                raise
        else:
            logging.info(f"Using existing reference audio: {reference_output_path}")
        
        # Extract test audio from reference
        logging.info("Extracting test portion from reference audio...")
        test_audio, sr = extract_test_audio(reference_output_path)
        
        if test_audio is None:
            logging.error("Failed to extract test audio")
            return
        
        # Apply modifications
        logging.info("Applying voice modifications...")
        modified_audio = test_audio.copy()
        
        if args.pitch != 0.0:
            logging.info(f"Modifying pitch by {args.pitch} semitones...")
            modified_audio = modify_pitch(modified_audio, sr, args.pitch)
        
        if args.speed != 1.0:
            logging.info(f"Modifying speed by factor {args.speed}...")
            modified_audio = modify_speed(modified_audio, sr, args.speed)
        
        if args.brightness != 0.0:
            logging.info(f"Modifying brightness/timbre by {args.brightness}...")
            modified_audio = modify_timbre(modified_audio, sr, args.brightness)
        
        if args.expressiveness != 0.0:
            logging.info(f"Modifying expressiveness by {args.expressiveness}...")
            modified_audio = modify_expressiveness(modified_audio, sr, args.expressiveness)
        
        # Save the modified audio
        modified_output_path = os.path.join(args.output_dir, "cloned_modified.wav")
        save_audio_file(modified_audio, sr, modified_output_path)
        
        # Create variants if requested
        if args.create_variants:
            logging.info("Creating voice variants...")
            variant_params = [
                {"name": "higher_pitch", "pitch": 1.0, "speed": 1.0, "brightness": 0.0, "expressiveness": 0.0},
                {"name": "lower_pitch", "pitch": -1.0, "speed": 1.0, "brightness": 0.0, "expressiveness": 0.0},
                {"name": "faster", "pitch": 0.0, "speed": 1.1, "brightness": 0.0, "expressiveness": 0.0},
                {"name": "slower", "pitch": 0.0, "speed": 0.9, "brightness": 0.0, "expressiveness": 0.0},
                {"name": "brighter", "pitch": 0.0, "speed": 1.0, "brightness": 0.3, "expressiveness": 0.0},
                {"name": "darker", "pitch": 0.0, "speed": 1.0, "brightness": -0.3, "expressiveness": 0.0},
                {"name": "expressive", "pitch": 0.0, "speed": 1.0, "brightness": 0.0, "expressiveness": 0.3},
                {"name": "combined_1", "pitch": 0.7, "speed": 1.05, "brightness": 0.2, "expressiveness": 0.1},
                {"name": "combined_2", "pitch": -0.5, "speed": 0.95, "brightness": -0.1, "expressiveness": 0.2}
            ]
            
            for params in variant_params:
                logging.info(f"Creating variant: {params['name']}")
                try:
                    variant_audio = test_audio.copy()
                    
                    # Apply modifications
                    if params["pitch"] != 0.0:
                        variant_audio = modify_pitch(variant_audio, sr, params["pitch"])
                    
                    if params["speed"] != 1.0:
                        variant_audio = modify_speed(variant_audio, sr, params["speed"])
                    
                    if params["brightness"] != 0.0:
                        variant_audio = modify_timbre(variant_audio, sr, params["brightness"])
                    
                    if params["expressiveness"] != 0.0:
                        variant_audio = modify_expressiveness(variant_audio, sr, params["expressiveness"])
                    
                    # Save variant
                    variant_path = os.path.join(args.output_dir, f"variant_{params['name']}.wav")
                    save_audio_file(variant_audio, sr, variant_path)
                
                except Exception as e:
                    logging.error(f"Error creating variant {params['name']}: {e}")
            
            logging.info("Voice variants created successfully")
        
        logging.info("Voice cloning and modification completed successfully")
        
    except Exception as e:
        logging.error(f"Error in voice cloning process: {e}")

if __name__ == "__main__":
    main()