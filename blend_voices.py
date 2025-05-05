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


# ======== New addtional blending methods start ========

def improved_spectral_blend(audio1, audio2, sr, ratio=0.5):
    """Blend two voices in the spectral domain for more natural blending"""
    import numpy as np
    import librosa
    
    # Compute Short-Time Fourier Transform (STFT) for both audio files
    stft1 = librosa.stft(audio1)
    stft2 = librosa.stft(audio2)
    
    # Get magnitude and phase
    mag1, phase1 = librosa.magphase(stft1)
    mag2, phase2 = librosa.magphase(stft2)
    
    # Blend magnitudes and phases
    blended_mag = (1 - ratio) * mag1 + ratio * mag2
    
    # For phase, we'll use a weighted approach based on magnitude
    # This helps preserve the dominant characteristics
    mask = mag1 > mag2
    blended_phase = np.zeros_like(phase1, dtype=complex)
    blended_phase[mask] = phase1[mask]
    blended_phase[~mask] = phase2[~mask]
    
    # Combine blended magnitude and phase
    blended_stft = blended_mag * blended_phase
    
    # Inverse STFT to get the blended audio
    blended_audio = librosa.istft(blended_stft, length=len(audio1))
    
    return blended_audio

def crossfade_morph(audio1, audio2, sr, ratio=0.5, segment_length_ms=500):
    """
    Create a voice that morphs between two voices using segmented crossfades
    """
    import numpy as np
    
    # Convert segment length from ms to samples
    segment_samples = int(segment_length_ms * sr / 1000)
    
    # Ensure both audios are the same length
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    # Create output buffer
    blended_audio = np.zeros_like(audio1)
    
    # Number of segments
    num_segments = min_len // segment_samples
    
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        
        # Create crossfade window
        fade_in = np.linspace(0, 1, segment_samples)
        fade_out = np.linspace(1, 0, segment_samples)
        
        # Randomly select dominant voice for this segment, with bias based on ratio
        if np.random.random() < ratio:
            # Voice 2 dominant
            segment = audio2[start:end] * fade_out + audio1[start:end] * fade_in
        else:
            # Voice 1 dominant
            segment = audio1[start:end] * fade_out + audio2[start:end] * fade_in
        
        blended_audio[start:end] = segment
    
    # Handle any remaining samples
    if min_len % segment_samples > 0:
        start = num_segments * segment_samples
        # Just use weighted average for the last partial segment
        blended_audio[start:] = (1-ratio) * audio1[start:] + ratio * audio2[start:]
    
    return blended_audio

def formant_preserving_blend(audio1, audio2, sr, ratio=0.5):
    """Blend voices while attempting to preserve formant structures"""
    import numpy as np
    import librosa
    
    # Compute mel spectrograms
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    
    mel_spec1 = librosa.feature.melspectrogram(y=audio1, sr=sr, n_fft=n_fft, 
                                             hop_length=hop_length, n_mels=n_mels)
    mel_spec2 = librosa.feature.melspectrogram(y=audio2, sr=sr, n_fft=n_fft, 
                                             hop_length=hop_length, n_mels=n_mels)
    
    # Convert to log scale
    log_mel1 = librosa.power_to_db(mel_spec1)
    log_mel2 = librosa.power_to_db(mel_spec2)
    
    # Blend log mel spectrograms
    blended_log_mel = (1 - ratio) * log_mel1 + ratio * log_mel2
    
    # Convert back to linear scale
    blended_mel = librosa.db_to_power(blended_log_mel)
    
    # Attempt to reconstruct audio
    # This is an approximation; perfect reconstruction isn't possible with just mel specs
    blended_audio = librosa.feature.inverse.mel_to_audio(
        blended_mel, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    # Normalize
    blended_audio = librosa.util.normalize(blended_audio)
    
    return blended_audio



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
    parser.add_argument("--test-text", default="Hi. This is a test of the blended voice. Now tell me, how does it sound?", 
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
        
        # IMPROVED BLENDING APPROACH
        try:
            logging.info("Attempting improved voice blending with spectral techniques")
            
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
                    import librosa
                    raw_audio2 = librosa.resample(raw_audio2, orig_sr=sr2, target_sr=sr1)
                    sr2 = sr1
                
                # Trim to matching length
                min_len = min(len(raw_audio1), len(raw_audio2))
                raw_audio1 = raw_audio1[:min_len]
                raw_audio2 = raw_audio2[:min_len]
                
                # Import necessary libraries
                import librosa
                import numpy as np
                
                # Try different blending methods and save each
                
                # 1. Spectral blending
                blended_spectral = improved_spectral_blend(raw_audio1, raw_audio2, sr1, ratio=args.ratio)
                spectral_output_path = os.path.join(args.output_dir, "blended_voice_spectral.wav")
                sf.write(spectral_output_path, blended_spectral, sr1)
                logging.info(f"Spectral blended voice saved to {spectral_output_path}")
                
                # 2. Crossfade morphing
                blended_crossfade = crossfade_morph(raw_audio1, raw_audio2, sr1, ratio=args.ratio)
                crossfade_output_path = os.path.join(args.output_dir, "blended_voice_crossfade.wav")
                sf.write(crossfade_output_path, blended_crossfade, sr1)
                logging.info(f"Crossfade blended voice saved to {crossfade_output_path}")
                
                # 3. Formant-preserving blend (most natural sounding)
                try:
                    blended_formant = formant_preserving_blend(raw_audio1, raw_audio2, sr1, ratio=args.ratio)
                    formant_output_path = os.path.join(args.output_dir, "blended_voice_formant.wav")
                    sf.write(formant_output_path, blended_formant, sr1)
                    logging.info(f"Formant-preserving blended voice saved to {formant_output_path}")
                    
                    # Use the formant-preserving blend as the final output
                    final_output_path = os.path.join(args.output_dir, "blended_voice.wav")
                    sf.write(final_output_path, blended_formant, sr1)
                    logging.info(f"Final blended voice saved to {final_output_path}")
                except Exception as e:
                    logging.error(f"Error in formant-preserving blend: {e}")
                    # Fall back to spectral blending for final output
                    final_output_path = os.path.join(args.output_dir, "blended_voice.wav")
                    sf.write(final_output_path, blended_spectral, sr1)
                    logging.info(f"Final blended voice (fallback to spectral) saved to {final_output_path}")

        except Exception as e:
            logging.error(f"Error in improved blending approach: {e}")
            
    except Exception as e:
        logging.error(f"Error in voice blending process: {e}")

if __name__ == "__main__":
    main()