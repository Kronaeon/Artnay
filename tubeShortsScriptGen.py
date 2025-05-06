import os
import re
import logging
import json
import time
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YouTubeShortsScriptGenerator:
    """Generate scripts specifically formatted for YouTube Shorts."""
    
    def __init__(self, input_dir="CleanSC", output_dir="scripts", 
                 model_path="/home/horus/Projects/Models/gemma-3-finetune.Q8_0.gguf",
                 n_gpu_layers=35, n_ctx=2048, n_batch=512):
        """Initialize the YouTube Shorts Script Generator.
        
        Args:
            input_dir (str): Directory containing cleaned content files
            output_dir (str): Directory to save generated scripts
            model_path (str): Path to the GGUF model file for llama.cpp
            n_gpu_layers (int): Number of GPU layers to use for inference
            n_ctx (int): Context size for the model
            n_batch (int): Batch size for inference
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        
        # Create output directory if it doesn't exist
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize the LLaMA model for inference."""
        try:
            from llama_cpp import Llama
            logging.info(f"Initializing model: {self.model_path}")
            
            # Start timing
            start_time = time.time()
            
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                verbose=False
            )
            
            # Calculate load time
            load_time = time.time() - start_time
            logging.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def load_topic_data(self):
        """Load topic and content from topic_overview.md or individual files.
        
        Returns:
            tuple: (topic_name, consolidated_content)
        """
        # First check for a consolidated topic overview file
        overview_path = self.input_dir / "topic_overview.md"
        topic_name = ""
        consolidated_content = ""
        
        if overview_path.exists():
            logging.info(f"Loading topic overview from {overview_path}")
            content = self._read_file(overview_path)
            
            # Extract topic name from the first line if it follows # Topic Overview: [topic]
            topic_match = re.search(r'# Topic Overview: (.+)', content)
            if topic_match:
                topic_name = topic_match.group(1).strip()
            
            # Use the content after the first line
            content_lines = content.split('\n')[1:]
            consolidated_content = '\n'.join(content_lines).strip()
            
        else:
            # If no overview file, combine individual content files
            logging.info("No topic overview found. Combining individual content files.")
            content_files = list(self.input_dir.glob("clean_*.md"))
            
            if not content_files:
                raise FileNotFoundError(f"No content files found in {self.input_dir}")
            
            # Try to extract topic from filenames or content
            for file in content_files:
                content = self._read_file(file)
                
                # Look for YAML front matter
                yaml_match = re.search(r'---\n(.*?)\n---', content, re.DOTALL)
                if yaml_match:
                    yaml_content = yaml_match.group(1)
                    # Try to extract topic from metadata
                    topic_match = re.search(r'original_file: ".*?([^/]+?)\.txt"', yaml_content)
                    if topic_match and not topic_name:
                        topic_name = topic_match.group(1).replace('_', ' ').strip()
                
                # Extract content after YAML front matter
                content_match = re.search(r'---\n.*?\n---\n\n(.*)', content, re.DOTALL)
                if content_match:
                    file_content = content_match.group(1).strip()
                else:
                    file_content = content
                
                consolidated_content += file_content + "\n\n"
        
        # If still no topic name, use a default
        if not topic_name:
            topic_name = "YouTube Shorts Topic"
            
        logging.info(f"Loaded topic: {topic_name}")
        return topic_name, consolidated_content.strip()
    
    def create_script_prompt(self, topic, content, style="informative"):
        """Create a detailed prompt for the LLM with proper formatting requirements.
        
        Args:
            topic (str): The topic name
            content (str): The content to base the script on
            style (str): The style of the script (informative, entertaining, educational)
            
        Returns:
            str: The formatted prompt for the LLM
        """
        # Customize the prompt based on style
        style_instructions = {
            "informative": "Create an informative script that clearly explains key facts and insights about the topic.",
            "entertaining": "Create an entertaining script with humor and engaging language to capture viewers' attention.",
            "educational": "Create an educational script that teaches concepts in a structured, easy-to-understand way.",
            "controversial": "Create a script that presents a surprising or counterintuitive perspective on the topic."
        }
        
        style_instruction = style_instructions.get(style, style_instructions["informative"])
        
        prompt = f"""
You are an expert YouTube Shorts script writer. Your task is to create a script for a 60-second YouTube Short about "{topic}".

{style_instruction}

The script MUST follow this EXACT format:

```
# TITLE: [Catchy, keyword-rich title that will grab attention]

## HOOK
VISUAL: [Brief visual instructions for the first 3-5 seconds]
NARRATION: "[Hook text - an attention-grabbing opening line that makes viewers want to watch more]"

## MAIN CONTENT
VISUAL: [Visual instructions for the main content section]
NARRATION: "[Main content - Clear, concise explanations broken into 2-3 key points]"

## CONCLUSION
VISUAL: [Visual instructions for the conclusion]
NARRATION: "[Conclusion with a strong call-to-action]"
```

IMPORTANT REQUIREMENTS:
1. The script must be SHORT - maximum 150 words total for all narration combined
2. The script must be ENGAGING - hook viewers in the first 3 seconds
3. The script must be CLEAR - use simple language and short sentences
4. The script must include appropriate visual instructions alongside narration
5. The script must contain a hook, main content, and conclusion with call-to-action
6. The title should include relevant keywords for searchability

Here is the content to base your script on:
{content[:1500]}

Remember: YouTube Shorts are viewed vertically on mobile phones, so optimize your script for that format.
"""
        return prompt
    
    def generate_script(self, style="informative"):
        """Generate a script for a YouTube Short.
        
        Args:
            style (str): The style of the script (informative, entertaining, educational)
            
        Returns:
            dict: The parsed and validated script data
        """
        logging.info(f"Generating {style} script...")
        
        # Load topic data
        topic, content = self.load_topic_data()
        
        # Create prompt
        prompt = self.create_script_prompt(topic, content, style)
        
        # Generate script with the model
        start_time = time.time()
        try:
            response = self.llm.create_completion(
                prompt,
                max_tokens=1024,
                temperature=0.7,
                stop=[],
                stream=False
            )
            
            script_text = response["choices"][0]["text"]
            
            # Calculate generation time
            generation_time = time.time() - start_time
            logging.info(f"Script generated in {generation_time:.2f} seconds")
            
            # Parse and validate the script
            script_data = self.parse_script(script_text)
            validation_results = self.validate_script(script_data)
            
            # Add metadata
            script_data["topic"] = topic
            script_data["style"] = style
            script_data["generation_time"] = generation_time
            script_data["validation"] = validation_results
            
            # Save the script
            saved_path = self.save_script(script_data, style)
            script_data["file_path"] = str(saved_path)
            
            return script_data
            
        except Exception as e:
            logging.error(f"Error generating script: {e}")
            raise
    
    def parse_script(self, script_text):
        """Extract structured data from the generated text.
        
        Args:
            script_text (str): The raw generated script text
            
        Returns:
            dict: Structured script data
        """
        # Initialize script data
        script_data = {
            "title": "",
            "hook": {"visual": "", "narration": ""},
            "main_content": {"visual": "", "narration": ""},
            "conclusion": {"visual": "", "narration": ""},
            "raw_text": script_text
        }
        
        # Extract title
        title_match = re.search(r'# TITLE: (.+)', script_text)
        if title_match:
            script_data["title"] = title_match.group(1).strip()
        
        # Extract hook
        hook_section = re.search(r'## HOOK\s+VISUAL: ([^\n]+)\s+NARRATION: "([^"]+)"', script_text)
        if hook_section:
            script_data["hook"]["visual"] = hook_section.group(1).strip()
            script_data["hook"]["narration"] = hook_section.group(2).strip()
        
        # Extract main content
        main_section = re.search(r'## MAIN CONTENT\s+VISUAL: ([^\n]+)\s+NARRATION: "([^"]+)"', script_text)
        if main_section:
            script_data["main_content"]["visual"] = main_section.group(1).strip()
            script_data["main_content"]["narration"] = main_section.group(2).strip()
        
        # Extract conclusion
        conclusion_section = re.search(r'## CONCLUSION\s+VISUAL: ([^\n]+)\s+NARRATION: "([^"]+)"', script_text)
        if conclusion_section:
            script_data["conclusion"]["visual"] = conclusion_section.group(1).strip()
            script_data["conclusion"]["narration"] = conclusion_section.group(2).strip()
        
        return script_data
    
    def validate_script(self, script_data):
        """Check word count, estimated duration, segment presence.
        
        Args:
            script_data (dict): The parsed script data
            
        Returns:
            dict: Validation results
        """
        validation = {
            "issues": [],
            "word_counts": {},
            "estimated_duration": 0,
            "is_valid": True
        }
        
        # Check if all segments are present
        if not script_data["title"]:
            validation["issues"].append("Missing title")
            validation["is_valid"] = False
        
        if not script_data["hook"]["narration"]:
            validation["issues"].append("Missing hook narration")
            validation["is_valid"] = False
        
        if not script_data["main_content"]["narration"]:
            validation["issues"].append("Missing main content narration")
            validation["is_valid"] = False
        
        if not script_data["conclusion"]["narration"]:
            validation["issues"].append("Missing conclusion narration")
            validation["is_valid"] = False
        
        # Count words and calculate estimated duration
        hook_words = len(script_data["hook"]["narration"].split())
        main_words = len(script_data["main_content"]["narration"].split())
        conclusion_words = len(script_data["conclusion"]["narration"].split())
        total_words = hook_words + main_words + conclusion_words
        
        # Store word counts
        validation["word_counts"] = {
            "hook": hook_words,
            "main_content": main_words,
            "conclusion": conclusion_words,
            "total": total_words
        }
        
        # Calculate estimated duration (words ÷ 2.5 ≈ seconds)
        estimated_duration = total_words / 2.5
        validation["estimated_duration"] = estimated_duration
        
        # Check if script is within duration limits
        if estimated_duration > 60:
            validation["issues"].append(f"Script too long: estimated {estimated_duration:.1f} seconds (target: 60 seconds)")
            validation["is_valid"] = False
        
        # Check if hook is appropriately brief
        if hook_words > 25:
            validation["issues"].append(f"Hook too long: {hook_words} words (target: 10-15 words)")
        
        return validation
    
    def save_script(self, script_data, style="informative"):
        """Format and save the script to a file.
        
        Args:
            script_data (dict): The script data to save
            style (str): The style of the script
            
        Returns:
            Path: The path to the saved script file
        """
        topic = script_data.get("topic", "topic")
        
        # Create a clean filename
        clean_topic = re.sub(r'[^\w\s-]', '', topic).strip().lower()
        clean_topic = re.sub(r'[-\s]+', '-', clean_topic)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{clean_topic}-{style}-{timestamp}.md"
        
        filepath = self.output_dir / filename
        
        # Format the script for saving
        formatted_script = f"""---
title: "{script_data['title']}"
topic: "{topic}"
style: "{style}"
date: "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
word_count: {script_data['validation']['word_counts']['total']}
estimated_duration: {script_data['validation']['estimated_duration']:.1f}
validation: {json.dumps(script_data['validation']['issues'])}
---

# {script_data['title']}

## HOOK
**VISUAL:** {script_data['hook']['visual']}

**NARRATION:** "{script_data['hook']['narration']}"

## MAIN CONTENT
**VISUAL:** {script_data['main_content']['visual']}

**NARRATION:** "{script_data['main_content']['narration']}"

## CONCLUSION
**VISUAL:** {script_data['conclusion']['visual']}

**NARRATION:** "{script_data['conclusion']['narration']}"

---

### METADATA
- Word count (hook): {script_data['validation']['word_counts']['hook']}
- Word count (main): {script_data['validation']['word_counts']['main_content']}
- Word count (conclusion): {script_data['validation']['word_counts']['conclusion']}
- Total word count: {script_data['validation']['word_counts']['total']}
- Estimated duration: {script_data['validation']['estimated_duration']:.1f} seconds
"""

        # Save to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(formatted_script)
            logging.info(f"Script saved to {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"Error saving script: {e}")
            raise
    
    def _read_file(self, filepath):
        """Helper method to read a file.
        
        Args:
            filepath (Path): Path to the file
            
        Returns:
            str: The file content
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error reading file {filepath}: {e}")
            return ""
    
    def generate_multiple_styles(self, styles=None):
        """Generate scripts in multiple styles.
        
        Args:
            styles (list): List of styles to generate
            
        Returns:
            list: List of generated script data
        """
        if styles is None:
            styles = ["informative", "entertaining", "educational", "controversial"]
        
        results = []
        for style in styles:
            try:
                script_data = self.generate_script(style)
                results.append(script_data)
                logging.info(f"Generated {style} script: {script_data['title']}")
            except Exception as e:
                logging.error(f"Error generating {style} script: {e}")
        
        return results


# Example usage
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"YOUTUBE SHORTS SCRIPT GENERATOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Set environment variables if not already set
    if "LLAMA_CPP_LIB_PATH" not in os.environ:
        os.environ["LLAMA_CPP_LIB_PATH"] = "/home/horus/Workspace/llama.cpp/build/bin"
    
    try:
        # Initialize generator with GPU configuration
        generator = YouTubeShortsScriptGenerator(
            input_dir="CleanSC", 
            output_dir="scripts",
            model_path="/home/horus/Projects/Models/gemma-3-finetune.Q8_0.gguf",
            n_gpu_layers=35,  # Adjust based on GPU VRAM
            n_ctx=2048,
            n_batch=512
        )
        
        # Generate a script
        print("\nGenerating an informative script...")
        script_data = generator.generate_script(style="informative")
        
        # Print summary
        print("\nScript Summary:")
        print(f"Title: {script_data['title']}")
        print(f"Word count: {script_data['validation']['word_counts']['total']}")
        print(f"Estimated duration: {script_data['validation']['estimated_duration']:.1f} seconds")
        
        if script_data['validation']['issues']:
            print("\nValidation Issues:")
            for issue in script_data['validation']['issues']:
                print(f"- {issue}")
        
        print(f"\nScript saved to: {script_data['file_path']}")
        
        # Generate scripts in multiple styles
        print("\nWould you like to generate scripts in other styles? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            print("\nGenerating scripts in multiple styles...")
            results = generator.generate_multiple_styles()
            
            print("\nGenerated Scripts:")
            for result in results:
                print(f"- {result['style'].capitalize()}: {result['title']}")
        
    except Exception as e:
        print(f"\nError: {e}")
    
    print(f"\n{'='*60}")