from langchain_core.messages import HumanMessage
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.chat_models.base import BaseChatModel


class StructuredWebSearcherState(BaseModel):
    # Core configuration
    model: BaseChatModel # must support structured output with json_schema
    output_schema: Any # must be a pydantic model
    target_information: str # a description of the target information to be extracted
    queries: Optional[List[str]] = None # if None, the queries will be generated by the query generator

    # Flow Configuration
    summarize_scraped_contents: bool = True
    needs_formatting: bool = True
    scrape_search_results: bool = True
    max_retries: int = 3

    # Additional instructions
    additional_summarizer_instructions: str|None = None 
    additional_query_generator_instructions: str|None = None 
    additional_formatting_instructions: str|None = None

    # Search configuration
    num_queries: int = 1
    num_results_per_query: int = 10

    # Filtering configuration
    top_p_search_results: float = 1
    include_unofficial_sources: bool = True
    include_outdated_sources: bool = False
    prefer_official_sources: bool = True
    only_relevant_sources: bool = True

    # Scraping configuration
    scraping_batch_size: int = 10
    scraping_timeout: int = 20000
    max_cost_per_scrape: int = 10
    scrapingbee_concurrency_limit: int = 5
    return_scraped_screenshots: bool = True
    max_chars_per_scraped_source: int = 5000
    max_images_per_scraped_source: int = 10

    # Debugging
    verbose: bool = False

    # Internal state (should not be set by user)
    search_results: List[Dict[str, Any]] = []
    retries: int = 0
    all_scrapes_failed: bool = False
    failed_queries: List[str] = []
    relevant_search_results: List[Dict[str, Any]] = []
    formatted_search_results: List[Dict[str, Any]] = []
    summarized_content: Optional[str] = None
    target_information_source: Optional[HumanMessage] = None


    # Output (should not be set by user)
    output: BaseModel|Dict[str, Any] = None
    errors: List[str] = []



    

class Query(BaseModel):
    query: str


class QueryGeneratorOutputSchema(BaseModel):
    brainstorming_and_considerations: List[str]
    selected_and_refined_queries: List[Query]


class SearchResultRelevanceAnalysis(BaseModel):
    analysis: str = Field(..., description="a critical analysis of the search result to determine if it is relevant to the topic and the target information")
    is_official_source_for_target_information: bool = Field(..., description="whether the source is an official source for the target information")
    is_relevant_to_target_information: bool = Field(..., description="whether the source is relevant to the target information")
    is_well_known_comparison_site: bool = Field(..., description="whether the source is a well-known comparison site")
    relevance_score: int = Field(..., description="a score between 0 and 100 that represents the relevance of the search result to the topic and the target information based on the analysis")
    is_outdated: bool = Field(..., description="if it contains information that is directly contradicting an official source, set this to true else false")

    


