import time
import logging
import requests
from abc import ABC, abstractmethod
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RateLimiter:
    """Manages rate limiting by enforcing a delay between requests."""
    def __init__(self, calls_per_second=3.0):
        self.delay = 1.0 / calls_per_second
        self.last_request_time = 0

    def wait(self):
        """Ensures a delay has passed since the last request."""
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.delay:
            time.sleep(self.delay - time_since_last)
        self.last_request_time = time.time()

class SearchEngine(ABC):
    """Abstract base class defining the search engine interface."""
    @abstractmethod
    def search(self, query, max_results):
        """Searches for a query and returns a list of results."""
        pass

class BingSearchAPI(SearchEngine):
    """Search engine implementation for Bing using the official API.
    
    Attributes:
        api_key (str): Your Bing Search API key.
        rate_limiter (RateLimiter): Rate limiting instance.
        name (str): Name of the search engine.
    """
    def __init__(self, api_key, rate_limiter):
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.name = 'Bing API'
        self.endpoint = 'https://api.bing.microsoft.com/v7.0/search'

    def search(self, query, max_results):
        """Fetches search results from Bing API.

        Args:
            query (str): Search query.
            max_results (int): Maximum number of results to return.

        Returns:
            list: List of dicts with 'url', 'title', 'snippet', 'rank', 'search_engine'.
        """
        results = []
        offset = 0
        count = min(50, max_results)  # Bing API allows up to 50 results per request
        
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Accept': 'application/json'
        }
        
        while len(results) < max_results:
            params = {
                'q': query,
                'count': count,
                'offset': offset,
                'mkt': 'en-US',
                'responseFilter': 'Webpages'
            }
            
            self.rate_limiter.wait()
            
            try:
                response = requests.get(self.endpoint, headers=headers, params=params)
                response.raise_for_status()  # Raise an exception for non-200 status codes
                
                data = response.json()
                
                if 'webPages' not in data or 'value' not in data['webPages']:
                    logging.warning("No web pages found in Bing API response")
                    break
                
                pages = data['webPages']['value']
                
                if not pages:
                    break
                
                for i, page in enumerate(pages):
                    rank = len(results) + i + 1
                    results.append({
                        'url': page['url'],
                        'title': page['name'],
                        'snippet': page['snippet'],
                        'rank': rank,
                        'search_engine': self.name
                    })
                
                # Check if we need to get more results
                if len(pages) < count or len(results) >= max_results:
                    break
                    
                offset += count
                
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching Bing API results: {e}")
                break
                
        return results[:max_results]

class GoogleSearchAPI(SearchEngine):
    """Search engine implementation for Google using the Custom Search JSON API.
    
    Attributes:
        api_key (str): Your Google API key.
        cx (str): Your Custom Search Engine ID.
        rate_limiter (RateLimiter): Rate limiting instance.
        name (str): Name of the search engine.
    """
    def __init__(self, api_key, cx, rate_limiter):
        self.api_key = api_key
        self.cx = cx
        self.rate_limiter = rate_limiter
        self.name = 'Google API'
        self.endpoint = 'https://www.googleapis.com/customsearch/v1'

    def search(self, query, max_results):
        """Fetches search results from Google Custom Search API.

        Args:
            query (str): Search query.
            max_results (int): Maximum number of results to return.

        Returns:
            list: List of dicts with 'url', 'title', 'snippet', 'rank', 'search_engine'.
        """
        results = []
        start_index = 1
        
        # Google API allows only 10 results per request
        # and a maximum of 100 results per day on the free tier
        while len(results) < max_results:
            params = {
                'q': query,
                'key': self.api_key,
                'cx': self.cx,
                'start': start_index
            }
            
            self.rate_limiter.wait()
            
            try:
                response = requests.get(self.endpoint, params=params)
                response.raise_for_status()  # Raise an exception for non-200 status codes
                
                data = response.json()
                
                if 'items' not in data:
                    logging.warning("No items found in Google API response")
                    break
                
                items = data['items']
                
                if not items:
                    break
                
                for i, item in enumerate(items):
                    rank = start_index + i
                    results.append({
                        'url': item['link'],
                        'title': item['title'],
                        'snippet': item.get('snippet', ''),
                        'rank': rank,
                        'search_engine': self.name
                    })
                
                # Check if we need to get more results
                if len(items) < 10 or len(results) >= max_results:
                    break
                    
                start_index += 10
                
                # Google's free tier only allows 10 pages (100 results)
                if start_index > 100:
                    break
                
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching Google API results: {e}")
                break
                
        return results[:max_results]

class SearchQueryEngine:
    """Main class to handle search queries and filter results.

    Attributes:
        search_engines (list): List of SearchEngine instances.
        rate_limiter (RateLimiter): Rate limiting instance.
    """
    def __init__(self, search_engines, rate_limiter):
        self.search_engines = search_engines
        self.rate_limiter = rate_limiter

    def get_results(self, query, keywords_include=None, keywords_exclude=None, domains=None, num_results=10):
        """Fetches and filters search results.

        Args:
            query (str): Search query.
            keywords_include (list, optional): Keywords that must be present.
            keywords_exclude (list, optional): Keywords to exclude.
            domains (list, optional): Allowed domains.
            num_results (int): Number of results desired.

        Returns:
            list: Filtered list of result dictionaries.
        """
        if not query:
            raise ValueError("Query cannot be empty")
        
        keywords_include = keywords_include or []
        keywords_exclude = keywords_exclude or []
        domains = domains or []
        
        logging.info(f"Searching for: '{query}'")
        logging.info(f"Include keywords: {keywords_include}")
        logging.info(f"Exclude keywords: {keywords_exclude}")
        logging.info(f"Filter domains: {domains}")
        
        all_results = []
        for se in self.search_engines:
            logging.info(f"Using search engine: {se.name}")
            results = se.search(query, num_results)
            logging.info(f"Got {len(results)} raw results from {se.name}")
            all_results.extend(results)

        # Deduplicate by URL, keeping the best rank
        logging.info(f"Deduplicating {len(all_results)} total results")
        url_to_result = {}
        for result in all_results:
            url = result['url']
            if url not in url_to_result or result['rank'] < url_to_result[url]['rank']:
                url_to_result[url] = result

        unique_results = list(url_to_result.values())
        unique_results.sort(key=lambda x: x['rank'])
        logging.info(f"Found {len(unique_results)} unique results")

        # Apply filters
        filtered_results = []
        for result in unique_results:
            text = (result['title'] + ' ' + result['snippet']).lower()
            
            # Check keyword includes
            include_match = True
            for k in keywords_include:
                if k.lower() not in text:
                    include_match = False
                    break
                    
            # Check keyword excludes
            exclude_match = False
            for k in keywords_exclude:
                if k.lower() in text:
                    exclude_match = True
                    break
                    
            # Check domains
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
                filtered_results.append(result)
                logging.debug(f"Result passed filters: {result['title']}")
            else:
                reasons = []
                if not include_match:
                    reasons.append("missing required keywords")
                if exclude_match:
                    reasons.append("contains excluded keywords")
                if not domain_match:
                    reasons.append("domain not in allowed list")
                logging.debug(f"Result filtered out: {result['title']} - {', '.join(reasons)}")
                
        logging.info(f"After filtering: {len(filtered_results)} results")
        return filtered_results[:num_results]

# Example usage
if __name__ == "__main__":
    # Replace these with your actual API keys
    BING_API_KEY = "your_bing_api_key_here"
    GOOGLE_API_KEY = "your_google_api_key_here"
    GOOGLE_CX = "your_google_custom_search_engine_id_here"
    
    rate_limiter = RateLimiter(calls_per_second=1.0)  # Conservative rate limiting
    
    # You can use either one or both APIs
    search_engines = [
        BingSearchAPI(BING_API_KEY, rate_limiter),
        # GoogleSearchAPI(GOOGLE_API_KEY, GOOGLE_CX, rate_limiter)  # Uncomment to use Google as well
    ]
    
    engine = SearchQueryEngine(search_engines, rate_limiter)

    # Search without any filters
    results = engine.get_results(
        query="python programming",
        num_results=5
    )

    print(f"\nFound {len(results)} results:")
    for r in results:
        print(f"Rank: {r['rank']}, Engine: {r['search_engine']}, URL: {r['url']}, Title: {r['title']}")
        print(f"Snippet: {r['snippet'][:100]}...")
        print()

    # Search with filters
    results = engine.get_results(
        query="python programming",
        keywords_include=["tutorial"],
        keywords_exclude=["advertisement"],
        domains=["python.org", "tutorialspoint.com"],
        num_results=5
    )

    print(f"\nFound {len(results)} filtered results:")
    for r in results:
        print(f"Rank: {r['rank']}, Engine: {r['search_engine']}, URL: {r['url']}, Title: {r['title']}")
        print(f"Snippet: {r['snippet'][:100]}...")
        print()