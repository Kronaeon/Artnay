import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import llama_cpp

logging.basicConfig(level=logging.INFO)

def setup_local_llm(model_name="deepseek-ai/deepseek-coder-6.7b-base"):
    """Set up a local LLM for content processing.
    
    This function loads a model that's suitable for running on consumer GPUs
    like RTX 4060 Ti or RTX 4080.
    
    Args:
        model_name: HuggingFace model path
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Check GPU availability
    if torch.cuda.is_available():
        device = "cuda"
        gpu_info = torch.cuda.get_device_properties(0)
        logging.info(f"Using GPU: {gpu_info.name} with {gpu_info.total_memory / 1e9:.2f} GB memory")
    else:
        device = "cpu"
        logging.info("CUDA not available, using CPU (this will be very slow)")
    
    # Load tokenizer
    logging.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with optimizations
    logging.info(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None
    )
    
    logging.info("Model loaded successfully")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=1500):
    """Generate a response from the local model.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt: Input prompt text
        max_length: Maximum number of tokens to generate
        
    Returns:
        str: Generated text response
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Memory optimization for generation
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_length,
            temperature=0.1,
            do_sample=False,
            # Additional parameters for efficient generation
            use_cache=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the response if needed
    if prompt in response:
        response = response.replace(prompt, "")
    
    return response.strip()

# Example usage
if __name__ == "__main__":
    # Model options suitable for RTX 4060 Ti / 4080:
    
    # 1. DeepSeek models
    # model_name = "deepseek-ai/deepseek-coder-6.7b-base"  # Good for code understanding
    
    # 2. Smaller but effective options
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # Excellent performance for 7B
    # model_name = "microsoft/phi-2"  # Surprisingly good at 2.7B
    
    # 3. Using GGUF quantized models (lowest memory usage)
    # Install llama-cpp-python first:
    # pip install llama-cpp-python
    # model_name = "TheBloke/deepseek-coder-6.7b-instruct-GGUF"  # 4-bit quantization
    
    # Choose one model from above
    model_name = "deepseek-ai/deepseek-coder-6.7b-base"
    
    # Load model
    model, tokenizer = setup_local_llm(model_name)
    
    # Test with a simple prompt
    test_prompt = "Summarize the following text about AI in healthcare: [your text here]"
    response = generate_response(model, tokenizer, test_prompt)
    print(response)