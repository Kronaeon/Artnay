import argparse
import logging
from pathlib import Path
from dia.model import Dia

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Voice Cloning with Dia")
    parser.add_argument("voice_wav", help="Path to voice WAV file")
    parser.add_argument("voice_txt", help="Path to voice transcript TXT file")
    parser.add_argument("--output", default="voice_output.wav", help="Output audio file")
    parser.add_argument("--test-text", default="This is a test of voice cloning with the Dia model.", 
                      help="Text to synthesize with cloned voice")
    parser.add_argument("--model-path", default="/home/horus/Projects/Models/AUDIOMODELS/nari-labs",
                      help="Path to Dia model")
    
    args = parser.parse_args()
    
    try:
        # Load model
        logging.info(f"Loading Dia model from {args.model_path}")
        model = Dia.from_pretrained(args.model_path, compute_dtype="float16")
        
        # Load transcript
        with open(args.voice_txt, 'r') as f:
            voice_transcript = f.read().strip()
        
        # Format transcript with speaker tags if it doesn't already have them
        if not voice_transcript.startswith("[S"):
            voice_transcript = f"[S1] {voice_transcript}"
            
        # Format test text
        test_text = f"[S1] {args.test_text}"
        
        # Generate audio with voice cloning
        logging.info(f"Generating audio with cloned voice")
        logging.info(f"Using transcript: {voice_transcript}")
        logging.info(f"Generating text: {test_text}")
        
        # Generate using the approach from voice_clone.py example
        audio = model.generate(
            voice_transcript + test_text, 
            audio_prompt=args.voice_wav,
            use_torch_compile=True,
            verbose=True
        )
        
        # Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Use the model's save_audio method
        model.save_audio(str(output_path), audio)
        logging.info(f"Cloned voice audio saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Error generating voice: {e}")

if __name__ == "__main__":
    main()