import os
import re
import json
import logging
from pathlib import Path
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContentCleaner:
    """Clean scraped web content using either local LLM or API.
    
    This class processes raw scraped text from search results and:
    1. Removes boilerplate content, ads, navigation elements
    2. Extracts relevant information related to the topic
    3. Structures content in a clean, readable format
    4. Saves organized information to a new directory
    """
    
    def __init__(self, input_dir="search_content", output_dir="CleanSC", 
                 use_local_model=True, model_name="TheBloke/deepseek-coder-6.7b-base-GGUF",
                 api_endpoint=None, api_key=None):
        """Initialize the ContentCleaner.
        
        Args:
            input_dir (str): Directory containing scraped content files
            output_dir (str): Directory to save cleaned content
            use_local_model (bool): Whether to use a local LLM (True) or API (False)
            model_name (str): Name of the local model to use
            api_endpoint (str): API endpoint for remote LLM service
            api_key (str): API key for remote LLM service
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.use_local_model = use_local_model
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        
        # Create output directory if it doesn't exist
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            
        # Initialize model if using local
        if use_local_model:
            logging.info(f"Initializing local model: {model_name}")
            # Check for CUDA availability
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Using device: {self.device}")
            
            try:
                # Load model and tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="auto" if self.device == "cuda" else None
                )
                logging.info("Model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading model: {e}")
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
        """Clean content using a local LLM."""
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
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=1500,
                temperature=0.1,
                do_sample=False
            )
        
        # Decode and clean the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        if prompt in response:
            response = response.replace(prompt, "")
        
        return response.strip()
    
    def clean_with_api(self, content, topic):
        """Clean content using an API-based LLM."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        # CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES="75;89"" pip install llama-cpp-python
        # Prepare the API request
        data = {
            "model": "gpt-3.5-turbo",  # or your preferred model
            "messages": [
                {
                    "role": "system",
                    "content": "You are a content cleaning and information extraction assistant. Extract relevant, high-quality information from web content."
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
{content[:4000]}  # Truncating to avoid token limits

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
        # Create a filename based on the original
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
        
        # Read content
        content = self.read_file(filepath)
        if not content:
            logging.warning(f"Empty or unreadable content in {filepath}")
            return None
        
        # Extract metadata
        metadata = self.extract_metadata(content)
        
        # Clean content based on method
        if self.use_local_model:
            cleaned_content = self.clean_with_local_model(content, topic)
        else:
            cleaned_content = self.clean_with_api(content, topic)
        
        if not cleaned_content:
            logging.warning(f"No cleaned content generated for {filepath}")
            return None
        
        # Save cleaned content
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
        
        # Get all text files in input directory
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
        
        # Save summary
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
        
        logging.info(f"Processing complete. {summary_data['successful_files']} of {summary_data['total_files']} files processed successfully.")
        return processed_files
    
    def generate_topic_overview(self, topic, processed_files):
        """Generate a topic overview from all cleaned content files."""
        logging.info("Generating topic overview...")
        
        # Read all cleaned content
        all_cleaned_content = ""
        for file in processed_files:
            content = self.read_file(file)
            all_cleaned_content += content + "\n\n"
        
        # Create prompt for overview generation
        prompt = f"""
You are a research assistant tasked with creating a comprehensive topic overview.

Topic: {topic}

Based on the following cleaned research data, create a structured overview that:
1. Summarizes the key aspects of the topic
2. Identifies the main subtopics or themes
3. Organizes information in a logical sequence
4. Highlights important facts, trends, and insights

Here's the cleaned research data:
{all_cleaned_content[:6000]}  # Truncating to avoid token limits

Your overview should be comprehensive but focused, capturing the essence of the topic.
        """
        
        # Generate overview
        if self.use_local_model:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=2000,
                    temperature=0.2,
                    do_sample=True
                )
            overview = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in overview:
                overview = overview.replace(prompt, "")
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a research assistant tasked with creating comprehensive topic overviews."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = requests.post(self.api_endpoint, headers=headers, json=data)
            result = response.json()
            overview = result["choices"][0]["message"]["content"]
        
        # Save overview
        overview_path = self.output_dir / "topic_overview.md"
        with open(overview_path, 'w', encoding='utf-8') as f:
            f.write(f"# Topic Overview: {topic}\n\n")
            f.write(overview)
        
        logging.info(f"Topic overview generated and saved to {overview_path}")
        return overview_path


# Example usage
if __name__ == "__main__":
    # Set your topic - should match what was used for search
    TOPIC = "southpark cartoon, cartman" # "artificial intelligence in healthcare"
    
    # Option 1: Use local model (requires GPU)
    cleaner_local = ContentCleaner(
        input_dir="search_content",
        output_dir="CleanSC",
        use_local_model=True,
        model_name="TheBloke/deepseek-coder-6.7b-base-GGUF"  # Choose appropriate model
    )
    
    # Option 2: Use API (OpenAI example)
    # cleaner_api = ContentCleaner(
    #     input_dir="search_content",
    #     output_dir="CleanSC",
    #     use_local_model=False,
    #     api_endpoint="https://api.openai.com/v1/chat/completions",
    #     api_key="your-api-key"
    # )
    
    # Process files
    processed_files = cleaner_local.process_all_files(TOPIC)
    
    # Generate topic overview
    if processed_files:
        cleaner_local.generate_topic_overview(TOPIC, processed_files)