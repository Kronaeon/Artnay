import os
import re
import json
import logging
from pathlib import Path
import requests
from datetime import datetime
# from llama_cpp import Llama  # Import Llama from llama_cpp_python

# Set environment variables
os.environ["LLAMA_CPP_LIB_PATH"] = "/home/horus/Workspace/llama.cpp/build/bin"

# Print header with timestamp for tracking runs
print(f"\n{'='*60}")
print(f"MODEL INFERENCE TEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")

# Import library
print("Loading llama_cpp library...")
from llama_cpp import Llama


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContentCleaner:
    """Clean scraped web content using either local LLM (Llama.cpp) or API.
    
    This class processes raw scraped text from search results and:
    1. Removes boilerplate content, ads, navigation elements
    2. Extracts relevant information related to the topic
    3. Structures content in a clean, readable format
    4. Saves organized information to a new directory
    """
    
    def __init__(self, input_dir="search_content", output_dir="CleanSC", 
             use_local_model=True, model_path="meta-llama/Llama-3-8B-Instruct",
             api_endpoint=None, api_key=None, n_gpu_layers=35):  # Note: removed n_ctx, added n_gpu_layers
        """Initialize the ContentCleaner."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.use_local_model = use_local_model
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        
        # Create output directory if it doesn't exist
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            
        # Initialize llama.cpp if using local model
        if use_local_model:
            logging.info(f"Initializing llama-cpp-python with model: {model_path}")
            try:
                # Check GPU availability
                import torch
                if torch.cuda.is_available():
                    logging.info(f"CUDA is available. GPU count: {torch.cuda.device_count()}")
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        logging.info(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1e9:.1f} GB")
                else:
                    logging.warning("CUDA is NOT available!")
                
                # Initialize Llama.cpp with GPU support
                self.llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,  # This is the key parameter for GPU offloading
                    n_ctx=2048,                 # Context size
                    n_batch=512,                # Batch size
                    n_threads=8,                # Number of CPU threads
                    verbose=True                # Show loading details
                )
                
                logging.info("Model loaded successfully with GPU support")
                
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                raise
            
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
        """Clean content using a local LLM (Llama.cpp)."""
        # Create prompt for the model
        prompt = f"""
You are a content cleaning and information extraction assistant.
Your task is to extract relevant, high-quality information from web content.

The topic is: {topic}

Please analyze the following scraped web content and:
1. Remove all advertisements, navigation elements, footers, and irrelevant content
2. Extract only the information that is relevant to the topic
3. Structure the information in a clean, readable format
4. Remove any duplicated information
5. Organize facts and details in a logical sequence

Here is the scraped content:
{content[:4000]}  # Truncating to avoid token limits

Return ONLY the cleaned, relevant information without any additional commentary.
        """
        
        try:
            # Generate text using Llama.cpp
            output = self.llm(
                prompt,
                max_tokens=1500,
                temperature=0,  # Greedy decoding
                # stop=[]  # Stop at double newline or adjust as needed
            )
            response = output['choices'][0]['text']
            print(f"\n{'='*60}")
            print(f"Response: {response}")
            print(f"\n{'='*60}")
            return response.strip()
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
        
        prompt = f"""
You are a research assistant tasked with creating a comprehensive topic overview.

Topic: {topic}

Based on the following cleaned research data, create a structured overview that:
1. Summarizes the key aspects of the topic
2. Identifies the main subtopics or themes
3. Organizes information in a logical sequence
4. Highlights important facts, trends, and insights

Here's the cleaned research data:
{all_cleaned_content[:6000]}

Your overview should be comprehensive but focused, capturing the essence of the topic.
        """
        
        if self.use_local_model:
            try:
                output = self.llm(
                    prompt,
                    max_tokens=2000,
                    temperature=0.2,  # Slightly higher for creativity in overview
                    stop=["\n\n"]
                )
                overview = output['choices'][0]['text']
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
                    {"role": "system", "content": "You are a research assistant."},
                    {"role": "user", "content": prompt}
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
# if __name__ == "__main__":
#     TOPIC = "southpark cartoon, cartman"
    
#     cleaner = ContentCleaner(
#         input_dir="search_content",
#         output_dir="CleanSC",
#         use_local_model=True,
#         model_path="/home/horus/Projects/Models/gemma-3-finetune.Q8_0.gguf",  # Replace with actual GGUF file path
#         n_ctx=2048
#     )
    
#     processed_files = cleaner.process_all_files(TOPIC)
#     if processed_files:
#         cleaner.generate_topic_overview(TOPIC, processed_files)
        
        
# Example usage
if __name__ == "__main__":
    TOPIC = "southpark cartoon, cartman"
    
    cleaner = ContentCleaner(
        input_dir="search_content",
        output_dir="CleanSC",
        use_local_model=True,
        model_path="/home/horus/Projects/Models/gemma-3-finetune.Q8_0.gguf",
        n_gpu_layers=35  # Make sure this is passed correctly
    )
    
    processed_files = cleaner.process_all_files(TOPIC)
    if processed_files:
        cleaner.generate_topic_overview(TOPIC, processed_files)
        
        
# python -c "from llama_cpp import Llama; llm = Llama(model_path='/home/horus/Projects/Models/gemma-3-finetune.Q8_0.gguf'); output = llm('Hello, world!', max_tokens=10); echo $output"
# set -x LD_LIBRARY_PATH ~/Workspace/llama.cpp/build/bin /usr/local/cuda-12.8/lib64 $LD_LIBRARY_PATH
# set -x CMAKE_ARGS "-DLLAMA_BUILD=OFF -DGGML_CUDA=ON -DLLAMA_INCLUDE_DIR=$HOME/Workspace/llama.cpp/include -DLLAMA_LIBRARY=$HOME/Workspace/llama.cpp/build/bin/libllama.so"

