import os
from typing import List
from .schemas import StructuredWebSearcherState

def load_prompt(file_name: str) -> str:
    """Load prompt text from a file."""
    prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', file_name)
    with open(prompt_path, 'r', encoding='utf-8') as file:
        return file.read()
    

def get_next_urls_to_scrape(state: StructuredWebSearcherState, skip_urls: List[str]):
    search_results = state.relevant_search_results
    scraping_batch_size = state.scraping_batch_size
    
   
    # get the top p most relevant search results according to the score
    scrape_candidates = [result for result in search_results if result["link"] not in skip_urls]
    if state.prefer_official_sources: #order by official sources first and then by score
        scrape_candidates = sorted(scrape_candidates, key=lambda x: (x['official_source'], x['score']), reverse=True)
    else:
        scrape_candidates = sorted(scrape_candidates, key=lambda x: x['score'], reverse=True)

    relevant_search_results = scrape_candidates[:scraping_batch_size]
    return [result["link"] for result in relevant_search_results]

