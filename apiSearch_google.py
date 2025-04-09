# -----======================================================
# Author: M. Arthur Dean (Kronaeon)
# Created: 04/05/2025
# Last Update: 04/08/2025
# VD8931
# ProgramName: api Search
# Version: 1.2
# ------======================================================

import time
import logging
import requests
from urllib.parse import urlparse

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

    def filter_results(self, results, keywords_include=None, keywords_exclude=None, domains=None):
        """Filter search results based on various criteria.

        Args:
            results (list): List of search result dictionaries.
            keywords_include (list): Keywords that must be present in title or snippet.
            keywords_exclude (list): Keywords that must NOT be present in title or snippet.
            domains (list): List of domains to filter results by.

        Returns:
            list: Filtered list of search result dictionaries.
        """
        keywords_include = keywords_include or []
        keywords_exclude = keywords_exclude or []
        domains = domains or []
        
        filtered = []
        
        for result in results:
            # Combine title and snippet for keyword matching
            text = (result['title'] + ' ' + result['snippet']).lower()
            
            # Check if all required keywords are present
            include_match = all(k.lower() in text for k in keywords_include)
            
            # Check if any excluded keywords are present
            exclude_match = any(k.lower() in text for k in keywords_exclude)
            
            # Check domain restriction
            domain_match = True
            if domains:
                domain_match = False
                result_domain = urlparse(result['url']).netloc
                for d in domains:
                    if d in result_domain:
                        domain_match = True
                        break
            
            # Apply all filters
            if include_match and not exclude_match and domain_match:
                filtered.append(result)
                
        return filtered

class searchArea:
    def __init__(self, filename = "searchFile.txt"):
        self.topic = ""
        self.keyword_include = []
        self.keyword_exclude = []
        self.domains = []
        
        self.data = self.readFile(filename)
        
        
    
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
            print(f"Error: File '{filename}' not found.")
        except Exception as e:
            print(f"Error parsing file: {e}")



# Example usage
if __name__ == "__main__":
    # Replace with your actual API key and search engine ID
    API_KEY = ""
    CX = "5577aa179034e4ac0"
    
    # Create Search Area
    searchArea = searchArea()
    num_results = 10
    
    # Create search client
    search_client = GoogleCustomSearch(API_KEY, CX)
    
    
    
    # Basic search
    print("Basic Search:")
    
    results = search_client.search(searchArea.topic, num_results)
    
    for result in results:
        print(f"[{result['rank']}] {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Snippet: {result['snippet'][:100]}...")
        print()
    
    # Search with filters
    print("\nFiltered Search:")
    all_results = search_client.search(searchArea.topic, num_results=20)
    filtered_results = search_client.filter_results(
        all_results,
        searchArea.keyword_include,
        searchArea.keyword_exclude,
        searchArea.domains
    )
    
    print(f"Found {len(filtered_results)} results after filtering:")
    for result in filtered_results:
        print(f"[{result['rank']}] {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Snippet: {result['snippet'][:100]}...")
        print()
        
        
