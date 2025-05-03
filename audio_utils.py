from typing import Dict, List, Optional, Tuple, Any


import os
import json
import time
import wave
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from audio_generator import AudioGenerator

class AudioUtils:
    """Utility functions for audio generation and testing."""
    
    @staticmethod
    def benchmark_model(model_name: str, 
                       test_text: str = "This is a test of the audio generation system.",
                       iterations: int = 3) -> Dict[str, float]:
        """
        Benchmark a TTS model's performance.
        
        Args:
            model_name: Name of the model to benchmark
            test_text: Text to use for testing
            iterations: Number of test iterations
            
        Returns:
            Dictionary with performance metrics
        """
        results = {
            "model": model_name,
            "avg_generation_time": 0.0,
            "text_length": len(test_text),
            "words_per_second": 0.0,
            "total_time": 0.0
        }
        
        try:
            generator = AudioGenerator(model_name=model_name)
            times = []
            
            for i in range(iterations):
                start_time = time.time()
                audio_file = generator._generate_audio(f"benchmark_{i}", test_text)
                end_time = time.time()
                
                if audio_file and audio_file.exists():
                    times.append(end_time - start_time)
                    # Clean up test file
                    audio_file.unlink()
            
            if times:
                avg_time = sum(times) / len(times)
                words = len(test_text.split())
                wps = words / avg_time if avg_time > 0 else 0
                
                results.update({
                    "avg_generation_time": avg_time,
                    "words_per_second": wps,
                    "total_time": sum(times)
                })
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    @staticmethod
    def visualize_audio(audio_file: str, output_image: str = None) -> None:
        """
        Create visualization of audio waveform and spectrogram.
        
        Args:
            audio_file: Path to audio file
            output_image: Path to save visualization (optional)
        """
        y, sr = librosa.load(audio_file)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Waveform
        ax1.plot(y)
        ax1.set_title('Waveform')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Amplitude')
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax2)
        ax2.set_title('Spectrogram')
        fig.colorbar(img, ax=ax2, format='%+2.0f dB')
        
        plt.tight_layout()
        
        if output_image:
            plt.savefig(output_image, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    @staticmethod
    def analyze_audio_quality(audio_file: str) -> Dict[str, float]:
        """
        Analyze audio quality metrics.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary with audio quality metrics
        """
        y, sr = librosa.load(audio_file)
        
        metrics = {
            "sample_rate": sr,
            "duration": len(y) / sr,
            "rms_energy": np.sqrt(np.mean(y**2)),
            "peak_amplitude": np.max(np.abs(y)),
            "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)[0]),
            "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
        }
        
        return metrics
    
    @staticmethod
    def create_voice_profile(voice_sample: str) -> Dict[str, Any]:
        """
        Create a voice profile from a sample.
        
        Args:
            voice_sample: Path to voice sample file
            
        Returns:
            Dictionary with voice characteristics
        """
        y, sr = librosa.load(voice_sample)
        
        # Extract voice features
        profile = {
            "sample_rate": sr,
            "duration": len(y) / sr,
            "pitch_mean": float(np.mean(librosa.yin(y, fmin=50, fmax=400))),
            "pitch_std": float(np.std(librosa.yin(y, fmin=50, fmax=400))),
            "spectral_centroid_mean": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])),
            "energy_mean": float(np.mean(librosa.feature.rms(y=y)[0])),
            "tempo": float(librosa.beat.tempo(y=y, sr=sr)[0])
        }
        
        return profile
    
    @staticmethod
    def normalize_audio(input_file: str, output_file: str = None, target_db: float = -20.0) -> str:
        """
        Normalize audio to target loudness.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to save normalized audio (optional)
            target_db: Target peak dB level
            
        Returns:
            Path to normalized audio file
        """
        y, sr = librosa.load(input_file)
        
        # Calculate current peak dB
        current_peak_db = 20 * np.log10(np.max(np.abs(y)))
        
        # Calculate scaling factor
        scaling_factor = 10 ** ((target_db - current_peak_db) / 20)
        
        # Apply normalization
        y_normalized = y * scaling_factor
        
        # Save normalized audio
        if output_file is None:
            output_file = input_file.replace('.wav', '_normalized.wav')
        
        sf.write(output_file, y_normalized, sr)
        
        return output_file
    
    @staticmethod
    def concatenate_audio_files(audio_files: List[str], output_file: str, 
                               crossfade_duration: float = 0.5) -> str:
        """
        Concatenate multiple audio files with crossfade.
        
        Args:
            audio_files: List of audio file paths
            output_file: Path to save concatenated audio
            crossfade_duration: Duration of crossfade in seconds
            
        Returns:
            Path to concatenated audio file
        """
        # Load all audio files
        audio_data = []
        sample_rates = []
        
        for file in audio_files:
            y, sr = librosa.load(file)
            audio_data.append(y)
            sample_rates.append(sr)
        
        # Check if all files have the same sample rate
        if len(set(sample_rates)) > 1:
            # Resample all to highest sample rate
            target_sr = max(sample_rates)
            audio_data = [librosa.resample(y, orig_sr=sr, target_sr=target_sr) 
                         for y, sr in zip(audio_data, sample_rates)]
        else:
            target_sr = sample_rates[0]
        
        # Concatenate with crossfade
        final_audio = []
        crossfade_samples = int(crossfade_duration * target_sr)
        
        for i, audio in enumerate(audio_data):
            if i == 0:
                final_audio.extend(audio)
            else:
                # Apply crossfade
                prev_end = final_audio[-crossfade_samples:]
                curr_start = audio[:crossfade_samples]
                
                # Create crossfade
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)
                
                crossfaded = prev_end * fade_out + curr_start * fade_in
                
                # Replace end of previous audio with crossfade
                final_audio[-crossfade_samples:] = crossfaded
                
                # Add remaining part of current audio
                final_audio.extend(audio[crossfade_samples:])
        
        # Save concatenated audio
        sf.write(output_file, np.array(final_audio), target_sr)
        
        return output_file
    
    @staticmethod
    def test_all_models(test_text: str = "Testing all text-to-speech models.") -> Dict[str, Dict]:
        """
        Test all available TTS models and compare performance.
        
        Args:
            test_text: Text to use for testing
            
        Returns:
            Dictionary with results for each model
        """
        results = {}
        available_models = AudioGenerator.list_available_models()
        
        for model_name in available_models:
            print(f"Testing {model_name}...")
            results[model_name] = AudioUtils.benchmark_model(model_name, test_text)
        
        return results
    
    @staticmethod
    def generate_model_comparison_report(results: Dict[str, Dict], output_file: str = "model_comparison.json") -> None:
        """
        Generate a comprehensive model comparison report.
        
        Args:
            results: Dictionary with model benchmark results
            output_file: Path to save the report
        """
        # Calculate relative performance
        if results:
            # Find fastest and slowest models
            fastest_time = min(r.get("avg_generation_time", float('inf')) for r in results.values() if "avg_generation_time" in r)
            
            for model, result in results.items():
                if "avg_generation_time" in result:
                    result["relative_speed"] = fastest_time / result["avg_generation_time"]
                    result["performance_rank"] = None  # Will be filled later
            
            # Rank models by performance
            models_sorted = sorted([m for m in results.items() if "avg_generation_time" in m[1]], 
                                 key=lambda x: x[1]["avg_generation_time"])
            
            for rank, (model, _) in enumerate(models_sorted, 1):
                results[model]["performance_rank"] = rank
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Generate summary visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        models = []
        times = []
        speeds = []
        
        for model, result in results.items():
            if "avg_generation_time" in result:
                models.append(model)
                times.append(result["avg_generation_time"])
                speeds.append(result.get("words_per_second", 0))
        
        x = range(len(models))
        
        # Bar chart for generation time
        bars = ax.bar(x, times, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Generation Time (seconds)')
        ax.set_title('TTS Model Performance Comparison')
        
        # Add speed labels on bars
        for i, (bar, speed) in enumerate(zip(bars, speeds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{speed:.1f} WPS',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_file.replace('.json', '_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def create_test_script() -> str:
        """
        Create a test script file for audio generation testing.
        
        Returns:
            Path to the test script file
        """
        test_script_content = """---
title: "Test Script for Audio Generation"
topic: "Testing TTS Models"
style: "test"
date: "2025-01-02 12:00:00"
word_count: 25
estimated_duration: 15.0
validation: []
---

# Test Script for Audio Generation

## HOOK
**VISUAL:** Show text-to-speech animation
**NARRATION:** "This is a test of the audio generation system."

## MAIN CONTENT
**VISUAL:** Display model information
**NARRATION:** "We are testing multiple text-to-speech models for quality comparison."

## CONCLUSION
**VISUAL:** Show performance metrics
**NARRATION:** "Audio generation test completed successfully."
"""
        
        # Save test script
        test_script_path = Path("media_assets") / "test_script.md"
        test_script_path.parent.mkdir(exist_ok=True)
        
        with open(test_script_path, 'w') as f:
            f.write(test_script_content)
        
        return str(test_script_path)


# Example usage for testing
if __name__ == "__main__":
    # Create and benchmark all models
    test_results = AudioUtils.test_all_models()
    AudioUtils.generate_model_comparison_report(test_results)
    
    # Visualize a sample audio file
    # AudioUtils.visualize_audio("media_assets/audio/hook.wav", "waveform_visualization.png")
    
    # Create a test script and generate audio
    test_script = AudioUtils.create_test_script()
    
    # Test with a specific model
    generator = AudioGenerator(model_name="openvoice_v2")
    audio_files = generator.generate_narration(test_script)
    
    # Analyze the generated audio
    for audio_file in audio_files:
        if audio_file.exists():
            metrics = AudioUtils.analyze_audio_quality(str(audio_file))
            print(f"\nAudio metrics for {audio_file.name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")