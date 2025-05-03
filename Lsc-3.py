import os
import re
import json
import logging
import torch
from pathlib import Path
import requests
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContentCleaner:
    """Clean scraped web content using vLLM for efficient CUDA-accelerated inference.
    
    This class processes raw scraped text from search results and:
    1. Removes boilerplate content, ads, navigation elements
    2. Extracts relevant information related to the topic
    3. Structures content in a clean, readable format
    4. Saves organized information to a new directory
    """
    
    def __init__(self, input_dir="search_content", output_dir="CleanSC", 
                 use_local_model=True, model_path="meta-llama/Llama-3-8B-Instruct",
                 api_endpoint=None, api_key=None, tensor_parallel_size=1):
        """Initialize the ContentCleaner.
        
        Args:
            input_dir (str): Directory containing scraped content files
            output_dir (str): Directory to save cleaned content
            use_local_model (bool): Whether to use a local LLM (True) or API (False)
            model_path (str): Model name or path (HuggingFace format)
            api_endpoint (str): API endpoint for remote LLM service
            api_key (str): API key for remote LLM service
            tensor_parallel_size (int): Number of GPUs to use for tensor parallelism
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.use_local_model = use_local_model
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        
        # Create output directory if it doesn't exist
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            
        # Initialize vLLM if using local model
        if use_local_model:
            logging.info(f"Initializing vLLM with model: {model_path}")
            try:
                # Check GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logging.info(f"GPU Memory: {gpu_memory:.1f} GB")
                
                # Initialize vLLM with optimized settings for 16GB GPU
                # self.llm = LLM(
                #     model=model_path,
                #     gpu_memory_utilization=0.9,  # Use up to 90% of GPU memory
                #     max_model_len=4096,          # Adjust based on your needs
                #     tensor_parallel_size=tensor_parallel_size,
                #     enforce_eager=True,          # Better memory efficiency
                #     dtype="float16"              # Use FP16 for better performance
                # )
                
                # self.llm = LLM(
                #     model=model_path,
                #     gpu_memory_utilization=0.85,  # Slightly conservative
                #     max_model_len=4096,
                #     tensor_parallel_size=tensor_parallel_size,  # This will now be 2
                #     enforce_eager=True,
                #     dtype="float16"
                # )
                
                self.llm = LLM(
                    model=model_path,
                    gpu_memory_utilization=0.85,  # Reduce from 0.9 for better stability
                    max_model_len=4096,
                    tensor_parallel_size=tensor_parallel_size,  # Now will be 2
                    enforce_eager=True,
                    dtype="float16",
                    # distributed_init_method="auto"  # Add this line for auto GPU detection
                )
                
                # Set up sampling parameters
                self.sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=1500,
                    stop=["\n\n"],  # Stop on double newline
                    skip_special_tokens=True
                )
                
                # Initialize tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                logging.info("vLLM loaded successfully")
                
            except Exception as e:
                logging.error(f"Error loading vLLM: {e}")
                raise
        else:
            logging.info("Using API for content cleaning")
            if not api_endpoint or not api_key:
                raise ValueError("API endpoint and key are required when not using local model")
    
    def read_file(self, filepath):
        """Read content from a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error reading file {filepath}: {e}")
            return ""
    
    def extract_metadata(self, content):
        """Extract metadata from the content file."""
        metadata = {}
        lines = content.split('\n')
        
        for line in lines[:10]:  # Check first few lines for metadata
            if line.startswith("URL:"):
                metadata["url"] = line.replace("URL:", "").strip()
            elif line.startswith("Title:"):
                metadata["title"] = line.replace("Title:", "").strip()
            elif line.startswith("Rank:"):
                metadata["rank"] = line.replace("Rank:", "").strip()
                
        return metadata
    
    def clean_with_local_model(self, content, topic):
        """Clean content using vLLM for efficient CUDA inference."""
        
        # Create prompt for the model
        system_message = "You are a content cleaning and information extraction assistant."
        user_message = f"""Your task is to extract relevant, high-quality information from web content.

The topic is: {topic}

Please analyze the following scraped web content and:
1. Remove all advertisements, navigation elements, footers, and irrelevant content
2. Extract only the information that is relevant to the topic
3. Structure the information in a clean, readable format
4. Remove any duplicated information
5. Organize facts and details in a logical sequence

Here is the scraped content:
{content[:4000]}  # Truncating to avoid token limits

Return ONLY the cleaned, relevant information without any additional commentary."""
        
        # Format the prompt based on model type
        if "Llama-3" in self.llm.llm_engine.model_config.model:
            # Llama 3 format
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            # Generic format
            prompt = f"System: {system_message}\n\nUser: {user_message}\n\nAssistant:"
        
        try:
            # Generate text using vLLM
            outputs = self.llm.generate([prompt], self.sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            return response
            
        except Exception as e:
            logging.error(f"Error generating cleaned content: {e}")
            return ""
    
    def clean_with_api(self, content, topic):
        """Clean content using an API-based LLM."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a content cleaning and information extraction assistant."
                },
                {
                    "role": "user",
                    "content": f"""Please analyze the following scraped web content about '{topic}' and:
1. Remove all advertisements, navigation elements, footers, and irrelevant content
2. Extract only the information that is relevant to the topic
3. Structure the information in a clean, readable format
4. Remove any duplicated information
5. Organize facts and details in a logical sequence

Here is the scraped content:
{content[:4000]}

Return ONLY the cleaned, relevant information without any additional commentary."""
                }
            ]
        }
        
        try:
            response = requests.post(self.api_endpoint, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logging.error(f"API error: {e}")
            return f"Error cleaning content: {e}"
    
    def save_cleaned_content(self, filename, cleaned_content, metadata):
        """Save cleaned content to the output directory."""
        base_name = Path(filename).stem
        clean_filename = f"clean_{base_name}.md"
        output_path = self.output_dir / clean_filename
        
        # Add metadata as YAML front matter
        output_content = "---\n"
        for key, value in metadata.items():
            output_content += f"{key}: \"{value}\"\n"
        output_content += f"original_file: \"{filename}\"\n"
        output_content += "---\n\n"
        output_content += cleaned_content
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_content)
            logging.info(f"Saved cleaned content to {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Error saving cleaned content: {e}")
            return None
    
    def clean_file(self, filepath, topic):
        """Clean a single file and save the result."""
        logging.info(f"Processing file: {filepath}")
        
        content = self.read_file(filepath)
        if not content:
            logging.warning(f"Empty or unreadable content in {filepath}")
            return None
        
        metadata = self.extract_metadata(content)
        
        if self.use_local_model:
            cleaned_content = self.clean_with_local_model(content, topic)
        else:
            cleaned_content = self.clean_with_api(content, topic)
        
        if not cleaned_content:
            logging.warning(f"No cleaned content generated for {filepath}")
            return None
        
        return self.save_cleaned_content(filepath, cleaned_content, metadata)
    
    def process_all_files(self, topic):
        """Process all files in the input directory."""
        processed_files = []
        summary_data = {
            "topic": topic,
            "processed_files": [],
            "total_files": 0,
            "successful_files": 0
        }
        
        files = list(self.input_dir.glob("*.txt"))
        summary_data["total_files"] = len(files)
        
        for file in files:
            output_path = self.clean_file(file, topic)
            if output_path:
                summary_data["successful_files"] += 1
                summary_data["processed_files"].append({
                    "original": str(file),
                    "cleaned": str(output_path)
                })
                processed_files.append(output_path)
        
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
        
        logging.info(f"Processing complete. {summary_data['successful_files']} of {summary_data['total_files']} files processed successfully.")
        return processed_files
    
    def generate_topic_overview(self, topic, processed_files):
        """Generate a topic overview from all cleaned content files."""
        logging.info("Generating topic overview...")
        
        all_cleaned_content = ""
        for file in processed_files:
            content = self.read_file(file)
            all_cleaned_content += content + "\n\n"
        
        # Limit content to avoid token limit issues
        truncated_content = all_cleaned_content[:8000]  # Larger than before due to vLLM efficiency
        
        system_message = "You are a research assistant tasked with creating a comprehensive topic overview."
        user_message = f"""Topic: {topic}

Based on the following cleaned research data, create a structured overview that:
1. Summarizes the key aspects of the topic
2. Identifies the main subtopics or themes
3. Organizes information in a logical sequence
4. Highlights important facts, trends, and insights

Here's the cleaned research data:
{truncated_content}

Your overview should be comprehensive but focused, capturing the essence of the topic."""
        
        if self.use_local_model:
            try:
                # Format prompt
                if "Llama-3" in self.llm.llm_engine.model_config.model:
                    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                else:
                    prompt = f"System: {system_message}\n\nUser: {user_message}\n\nAssistant:"
                
                # Generate with slightly higher temperature for creativity
                sampling_params = SamplingParams(
                    temperature=0.2,
                    max_tokens=2000,
                    stop=["\n\n"],
                    skip_special_tokens=True
                )
                
                outputs = self.llm.generate([prompt], sampling_params)
                overview = outputs[0].outputs[0].text.strip()
                
            except Exception as e:
                logging.error(f"Error generating topic overview: {e}")
                overview = ""
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            }
            try:
                response = requests.post(self.api_endpoint, headers=headers, json=data)
                response.raise_for_status()
                overview = response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logging.error(f"API error: {e}")
                overview = f"Error generating overview: {e}"
        
        overview_path = self.output_dir / "topic_overview.md"
        with open(overview_path, 'w', encoding='utf-8') as f:
            f.write(f"# Topic Overview: {topic}\n\n")
            f.write(overview)
        
        logging.info(f"Topic overview saved to {overview_path}")
        return overview_path

# Example usage
if __name__ == "__main__":
    TOPIC = "southpark cartoon, cartman"
    
    # Initialize ContentCleaner with vLLM
    # cleaner = ContentCleaner(
    #     input_dir="search_content",
    #     output_dir="CleanSC",
    #     use_local_model=True,
    #     model_path="/home/horus/Projects/Models/VLLM/Llama-3-8B-Instruct",  # Example model
    #     tensor_parallel_size=1
    # )
    
# Modify your ContentCleaner initialization
    cleaner = ContentCleaner(
        input_dir="search_content",
        output_dir="CleanSC",
        use_local_model=True,
        model_path="/home/horus/Projects/Models/VLLM/Llama-3-8B-Instruct",
        tensor_parallel_size=2  # This tells vLLM to use 2 GPUs
    )
        
    processed_files = cleaner.process_all_files(TOPIC)
    if processed_files:
        cleaner.generate_topic_overview(TOPIC, processed_files)