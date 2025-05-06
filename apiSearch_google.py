# -----======================================================
# Author: M. Arthur Dean (Kronaeon)
# Created: 04/05/2025
# Last Update: 04/09/2025
# VD8931
# ProgramName: api Search
# Version: 1.3
# ------======================================================

import time
import logging
import requests
import os
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GoogleCustomSearch:
    """Google Custom Search API implementation.

    This class provides access to Google's Custom Search API, allowing you to search
    the web and retrieve structured results.

    Attributes:
        api_key (str): Your Google API key.
        cx (str): Your Custom Search Engine ID.
    """

    def __init__(self, api_key, cx):
        """Initialize the Google Custom Search.

        Args:
            api_key (str): Your Google API key.
            cx (str): Your Custom Search Engine ID.
        """
        self.api_key = api_key
        self.cx = cx
        self.endpoint = "https://www.googleapis.com/customsearch/v1"
        self.last_request_time = 0
        self.min_delay = 1.0  # Minimum delay between requests in seconds

    def _rate_limit(self):
        """Implement simple rate limiting to avoid API quota issues."""
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)
            
        self.last_request_time = time.time()

    def search(self, query, start=1, num_results=10, **kwargs):
        """Execute a search query and return the results.

        Args:
            query (str): The search query.
            start (int): The first result to retrieve (1-based indexing).
            num_results (int): Number of results to retrieve (max 10 per request).
            **kwargs: Additional parameters to pass to the API.

        Returns:
            list: A list of search result dictionaries.
        """
        results = []
        current_start = start
        total_needed = min(num_results, 100)  # API has a hard limit of 100 results (10 pages)
        
        while len(results) < total_needed:
            # Number of results to request in this batch (max 10 per API call)
            num = min(10, total_needed - len(results))
            
            self._rate_limit()
            
            try:
                params = {
                    'q': query,
                    'key': self.api_key,
                    'cx': self.cx,
                    'start': current_start,
                    'num': num
                }
                
                # Add any additional parameters
                params.update(kwargs)
                
                response = requests.get(self.endpoint, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # Check if we have search results
                if 'items' not in data:
                    logging.warning(f"No results found for query: {query}")
                    break
                
                # Process each search result
                for i, item in enumerate(data['items']):
                    rank = current_start + i
                    result = {
                        'rank': rank,
                        'url': item['link'],
                        'title': item['title'],
                        'snippet': item.get('snippet', ''),
                        'search_engine': 'Google CSE'
                    }
                    
                    # Add additional fields if available
                    if 'pagemap' in item and 'cse_thumbnail' in item['pagemap']:
                        result['thumbnail'] = item['pagemap']['cse_thumbnail'][0].get('src', '')
                    
                    results.append(result)
                
                # Check if we've reached the end of results
                if len(data['items']) < num:
                    break
                
                # Move to the next page
                current_start += num
                
                # Google API only allows up to 10 pages (100 results)
                if current_start > 100:
                    break
                    
            except requests.exceptions.RequestException as e:
                logging.error(f"Error accessing Google API: {e}")
                break
                
        return results[:num_results]

    # Deprecated, not currently working.
    def filter_results(self, results, keywords_include=None, keywords_exclude=None, domains=None, use_domains=False):
        keywords_include = keywords_include or []
        keywords_exclude = keywords_exclude or []
        domains = domains or []
        
        filtered = []
        
        for result in results:
            text = (result['title'] + ' ' + result['snippet']).lower()
            
            # Check if keywords_include is empty OR all keywords are present
            include_match = not keywords_include or all(k.lower() in text for k in keywords_include)
            
            # Check excluded keywords
            exclude_match = any(k.lower() in text for k in keywords_exclude)
            
            # Domain logic remains unchanged
            domain_match = True
            if use_domains and domains:
                domain_match = False
                result_domain = urlparse(result['url']).netloc
                for d in domains:
                    if d in result_domain:
                        domain_match = True
                        break
            
            if include_match and not exclude_match and domain_match:
                filtered.append(result)
            elif include_match and not exclude_match:
                print("Include and Exclude matched")
                filtered.append(result)
            elif include_match:
                print("Include matched")
                filtered.append(result)
            elif exclude_match:
                print("Exclude matched")
                filtered.append(result)
            else:
                print("No match")
                    
        return filtered


    def filterSearchQuery(self, topic, keywords_include=None, keywords_exclude=None, domains=None):
        
        for keyword in keywords_include:
            topic = topic + " " + '"' + keyword + '"'
        
        for keyword in keywords_exclude:
            topic = topic + " " + '-' + keyword
        
        return topic
        

    def fetch_page_content(self, url, timeout=10):
        """Fetch the content of a webpage.
        
        Args:
            url (str): The URL to fetch.
            timeout (int): Request timeout in seconds.
            
        Returns:
            str: The HTML content of the page, or None if the request failed.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching page content from {url}: {e}")
            return None
    
    def extract_text_from_html(self, html):
        """Extract main text content from HTML.
        
        Args:
            html (str): HTML content to parse.
            
        Returns:
            str: Extracted text content.
        """
        if not html:
            return ""
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
                
            # Get text
            text = soup.get_text(separator='\n')
            
            # Break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            
            # Remove blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logging.error(f"Error extracting text from HTML: {e}")
            return ""
    
    def save_content_to_file(self, result, content, output_dir="search_results"):
        """Save content to a text file.
        
        Args:
            result (dict): Search result dictionary.
            content (str): Text content to save.
            output_dir (str): Directory to save files in.
            
        Returns:
            str: Path to the saved file, or None if save failed.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create a filename based on the title or URL
        if result.get('title'):
            # Use title for filename but ensure it's valid
            base_filename = ''.join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in result['title'])
            base_filename = base_filename[:50]  # Limit length
        else:
            # Use a hash of the URL if no title
            base_filename = hashlib.md5(result['url'].encode()).hexdigest()
            
        # Add rank to ensure uniqueness
        filename = f"{result['rank']:03d}_{base_filename}.txt"
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Write URL and title as metadata at the top
                f.write(f"URL: {result['url']}\n")
                f.write(f"Title: {result['title']}\n")
                f.write(f"Rank: {result['rank']}\n")
                f.write("-" * 80 + "\n\n")
                
                # Write the actual content
                f.write(content)
                
            logging.info(f"Saved content to {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"Error saving content to file: {e}")
            return None
    
    def download_and_save_results(self, results, output_dir="search_results"):
        """Download content from search results and save to files.
        
        Args:
            results (list): List of search result dictionaries.
            output_dir (str): Directory to save files in.
            
        Returns:
            list: List of successful save paths.
        """
        saved_files = []
        
        for result in results:
            logging.info(f"Processing: [{result['rank']}] {result['title']} - {result['url']}")
            
            # Fetch HTML content
            html = self.fetch_page_content(result['url'])
            if not html:
                logging.warning(f"Could not fetch content from {result['url']}")
                continue
                
            # Extract text from HTML
            text_content = self.extract_text_from_html(html)
            if not text_content:
                logging.warning(f"Could not extract text from {result['url']}")
                continue
                
            # Save content to file
            filepath = self.save_content_to_file(result, text_content, output_dir)
            if filepath:
                saved_files.append(filepath)
                
            # Simple rate limiting to be nice to servers
            time.sleep(1)
            
        return saved_files

class SearchArea:
    def __init__(self, filename="searchFile.txt"):
        self.topic = ""
        self.keyword_include = []
        self.keyword_exclude = []
        self.domains = []
        
        self.readFile(filename)
    
    def readFile(self, filename):
        current_section = None
        
        try:
            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Check if this is a section header
                    if line.endswith(':'):
                        current_section = line[:-1].lower()  # Remove colon and convert to lowercase
                        continue
                    
                    # Add content to the appropriate section
                    if current_section == "topic":
                        self.topic = line
                    elif current_section == "keywordinclude":
                        self.keyword_include.append(line)
                    elif current_section == "keywordexclude":
                        self.keyword_exclude.append(line)
                    elif current_section == "domains":
                        self.domains.append(line)
        
        except FileNotFoundError:
            logging.error(f"Error: File '{filename}' not found.")
        except Exception as e:
            logging.error(f"Error parsing file: {e}")


# Example usage
if __name__ == "__main__":
    # Replace with your actual API key and search engine ID
    API_KEY = ""
    CX = "5577aa179034e4ac0"
    
    # Create Search Area
    search_area = SearchArea()
    num_results = 10
    output_dir = "search_content"
    
    # Create search client
    search_client = GoogleCustomSearch(API_KEY, CX)
    
    # Basic search
    print("Performing search for:", search_area.topic)
    results = search_client.search(search_area.topic, num_results=num_results)
    
    print(f"Found {len(results)} results:")
    for result in results:
        print(f"[{result['rank']}] {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Snippet: {result['snippet'][:100]}...")
        print()
    
    
    print("\n\nPerforming filtered search...\n\n")
    
    # Filtered search 2.0
    query = search_client.filterSearchQuery(search_area.topic, search_area.keyword_include, search_area.keyword_exclude, search_area.domains)
    results = search_client.search(query, num_results=num_results)
    
    
    # Apply filters if specified
    # if search_area.keyword_include or search_area.keyword_exclude or search_area.domains:
    #     print("\nApplying filters...")
    #     results = search_client.filter_results(
    #         results,
    #         search_area.keyword_include,
    #         search_area.keyword_exclude,
    #         search_area.domains
    #     )
        
    #     print(f"After filtering: {len(results)} results")
    
    # Download and save content
    if results:
        print(f"\nDownloading and saving content to '{output_dir}' directory...")
        saved_files = search_client.download_and_save_results(results, output_dir)
        
        print(f"\nSuccessfully saved {len(saved_files)} files:")
        for file in saved_files:
            print(f"- {file}")
    else:
        print("No results to download.")
        
