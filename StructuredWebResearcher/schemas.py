from langchain_core.messages import HumanMessage
from typing import TypedDict,List, Dict, Any
from pydantic import BaseModel, Field
from langchain.chat_models.base import BaseChatModel
from pydantic._internal._model_construction import ModelMetaclass
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from typing import Optional
from pydantic import fields


class StructuredWebSearcherState(BaseModel):

    queries: Optional[List[str]] = None
    failed_queries: List[str] = []
    retries: int = 0
    max_retries: int = 3
    target_information: str
    output_schema: Any
    needs_scraping: bool
    needs_summarization: bool
    max_cost_per_scrape: int

    model : BaseChatModel
    num_queries: int = 1

    top_p_search_results: float  = 1
    include_unofficial_sources: bool  = True
    include_outdated_sources: bool = False
    prefer_official_sources: bool = True
    only_relevant_sources: bool = True
    scraping_batch_size: int = 10
    scraping_timeout: int = 20000
    max_cost_per_scrape: int = 10
    scraping_batch_size: int = 10
    all_scrapes_failed: bool = False
    
    

    search_results: List[Dict[str, Any]] = []
    relevant_search_results: List[Dict[str, Any]] = []
    formatted_search_results: List[Dict[str, Any]] = []
    summarized_content: str = []
    output: BaseModel|Dict[str, Any] = None
    target_information_source: HumanMessage = None


    additional_summarizer_instructions: str|None
    additional_query_generator_instructions: str|None
    additional_formatting_instructions: str|None

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

    


