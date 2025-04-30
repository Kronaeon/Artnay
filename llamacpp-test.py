import os
import time
import json
from datetime import datetime

# Set environment variables
os.environ["LLAMA_CPP_LIB_PATH"] = "/home/horus/Workspace/llama.cpp/build/bin"

# Print header with timestamp for tracking runs
print(f"\n{'='*60}")
print(f"MODEL INFERENCE TEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")

# Import library
print("Loading llama_cpp library...")
from llama_cpp import Llama

# Model configuration
model_path = "/home/horus/Projects/Models/gemma-3-finetune.Q8_0.gguf"
prompt = "What is the composition of a rubber duck?"

# Display settings
config = {
    "model": model_path.split('/')[-1],
    "n_gpu_layers": 35,
    "n_ctx": 2048,
    "n_batch": 512,
    "temperature": 0.7,  # Add temperature control
    "max_tokens": 512    # Limit output length
}

print("\nCONFIGURATION:")
for k, v in config.items():
    print(f"  {k}: {v}")

# Load model with timing
print("\nLoading model...")
start_time = time.time()
try:
    model = Llama(
        model_path=model_path,
        n_gpu_layers=config["n_gpu_layers"],
        n_ctx=config["n_ctx"],
        n_batch=config["n_batch"],
        verbose=False  # Change to True for more detailed loading info
    )
    load_time = time.time() - start_time
    print(f"Model loaded successfully in {load_time:.2f} seconds")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Run inference with timing
print("\nGENERATING RESPONSE:")
print(f"Prompt: \"{prompt}\"")
print("\nThinking...", flush=True)

start_time = time.time()
try:
    # Use create_completion instead of generate for structured output
    response = model.create_completion(
        prompt,
        max_tokens=config["max_tokens"],
        temperature=config["temperature"],
        stream=False
    )
    inference_time = time.time() - start_time
    
    # Extract and format the response
    output_text = response["choices"][0]["text"]
    
    print("\nRESPONSE:")
    print(f"{output_text.strip()}")
    
    # Print performance stats
    tokens_generated = len(output_text.split())
    tokens_per_second = tokens_generated / inference_time
    
    print(f"\nPERFORMANCE:")
    print(f"  Generation time: {inference_time:.2f} seconds")
    print(f"  Tokens generated: ~{tokens_generated}")
    print(f"  Speed: ~{tokens_per_second:.2f} tokens/second")
    
except Exception as e:
    print(f"Error during inference: {e}")
    # Print traceback for debugging
    import traceback
    traceback.print_exc()

print(f"\n{'='*60}")