import argparse
import logging
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from dia.model import Dia

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Voice Blending with Dia")
    parser.add_argument("voice1_wav", help="Path to first voice WAV file")
    parser.add_argument("voice1_txt", help="Path to first voice transcript TXT file")
    parser.add_argument("voice2_wav", help="Path to second voice WAV file") 
    parser.add_argument("voice2_txt", help="Path to second voice transcript TXT file")
    parser.add_argument("--ratio", type=float, default=0.5, help="Blend ratio (0.0-1.0)")
    parser.add_argument("--output", default="blended_voice.wav", help="Output audio file")
    parser.add_argument("--test-text", default="This is a test of the blended voice. How does it sound?", 
                        help="Text to synthesize with blended voice")
    parser.add_argument("--model-path", default="/home/horus/Projects/Models/AUDIOMODELS/nari-labs",
                       help="Path to Dia model")
    
    args = parser.parse_args()
    
    try:
        # Load model
        logging.info(f"Loading Dia model from {args.model_path}")
        model = Dia.from_pretrained(args.model_path, compute_dtype="float16")
        
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
        
        # Generate audio with voice 1
        logging.info("Generating audio with voice 1")
        # Generate using the approach from voice_clone.py example
        audio1 = model.generate(
            voice1_transcript + test_text, 
            audio_prompt=args.voice1_wav,
            use_torch_compile=True,
            verbose=True
        )
        
        # Save voice 1 sample
        voice1_output = Path("voice1_sample.wav")
        sf.write(str(voice1_output), audio1, 44100)
        logging.info(f"Voice 1 sample saved to {voice1_output}")
        
        # Generate audio with voice 2
        logging.info("Generating audio with voice 2")
        audio2 = model.generate(
            voice2_transcript + test_text,
            audio_prompt=args.voice2_wav,
            use_torch_compile=True,
            verbose=True
        )
        
        # Save voice 2 sample
        voice2_output = Path("voice2_sample.wav")
        sf.write(str(voice2_output), audio2, 44100)
        logging.info(f"Voice 2 sample saved to {voice2_output}")
        
        # Blend the audio samples
        logging.info(f"Blending voices with ratio {args.ratio}")
        
        # Make sure both audio samples are the same length
        min_len = min(len(audio1), len(audio2))
        audio1_trimmed = audio1[:min_len]
        audio2_trimmed = audio2[:min_len]
        
        # Blend using the ratio
        blended_audio = (1 - args.ratio) * audio1_trimmed + args.ratio * audio2_trimmed
        
        # Save the blended audio
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        sf.write(str(output_path), blended_audio, 44100)
        logging.info(f"Blended voice saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Error blending voices: {e}")

if __name__ == "__main__":
    main()