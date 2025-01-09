from typing import TypedDict,List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
import re
from pydantic import BaseModel, create_model, Field
from .utils import load_prompt
from langchain.chat_models.base import BaseChatModel
from services.SerperSearchService import SerperSearchService
from services.ScrapingBeeService import ScrapingBeeService
import yaml
from utils.formatting_utils import format_sources
from utils.type_checking_utils import is_pydantic_model
import os
from urllib.parse import urlparse
from datetime import datetime

class StructuredWebResearcherInputSchema(TypedDict):
    target_information: str
    output_schema: BaseModel|Dict[str, Any]
    queries: List[str] | None = None

    needs_scraping: bool = True
    scraping_timeout: int = 20000
    needs_summarization: bool = True
    max_cost_per_scrape: int = 10
    scraping_batch_size: int = 10

    num_queries: int = 1
    model : BaseChatModel

    top_p_search_results: float = 1
    include_unofficial_sources: bool = True
    prefer_official_sources: bool = False
    only_relevant_sources: bool = False

    additional_summarizer_instructions: str|None = None
    additional_query_generator_instructions: str|None = None
    additional_formatting_instructions: str|None = None


class StructuredWebSearcherState(TypedDict):

    queries: List[str]
    target_information: str
    output_schema: BaseModel|Dict[str, Any]
    needs_scraping: bool
    needs_summarization: bool
    max_cost_per_scrape: int

    model : BaseChatModel
    num_queries: int

    top_p_search_results: float 
    include_unofficial_sources: bool 
    prefer_official_sources: bool
    only_relevant_sources: bool 
    scraping_batch_size: int
    scraping_timeout: int
    max_cost_per_scrape: int
    

    search_results: List[Dict[str, Any]]
    relevant_search_results: List[Dict[str, Any]]
    formatted_search_results: List[Dict[str, Any]]
    summarized_content: List[str]
    output: BaseModel|Dict[str, Any]
    target_information_source: HumanMessage


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

    

def generate_queries(state: StructuredWebSearcherState):
    print("🚩 generate_queries")

    query_generator_prompt = load_prompt("query_generator_prompt.txt")
    structured_model = state['model'].with_structured_output(QueryGeneratorOutputSchema, method="json_schema")

    response = structured_model.invoke([
        SystemMessage(content=query_generator_prompt),
        HumanMessage(content=yaml.dump({
            "user_input": state['target_information'],
            "num_queries": state['num_queries'],
            "additional_instructions": state['additional_query_generator_instructions'] if state['additional_query_generator_instructions'] else None
        }))
    ])
  
    queries = [query.query for query in response.selected_and_refined_queries]

    return {"queries": queries}



def execute_queries(state: StructuredWebSearcherState):
    print("🚩 execute_queries")
    serper_search_service = SerperSearchService()
    search_results = serper_search_service.search_google([{"q": state['queries'], "num": 10}])
    # TODO: remove check for prod
    if isinstance(search_results, dict):
        print(search_results.keys())

   
    return {"search_results": search_results}

def analyze_search_result_relevance(state: StructuredWebSearcherState):
    print("🚩 analyze_search_results")
    search_results = state['search_results']
    structured_output_dict = {}
    for search_result in search_results:
        property_name = re.sub(r'[^a-zA-Z0-9_]', '_', search_result['title'])
        search_result["property_name"] = property_name
        structured_output_dict[property_name] = (SearchResultRelevanceAnalysis, ...)
    print(structured_output_dict)
    dynamic_model = create_model("SearchResultsRelevanceAnalysis", **structured_output_dict)
    print(dynamic_model.model_json_schema())
    llm_with_structured_output = state['model'].with_structured_output(dynamic_model, method="json_schema")
    response = llm_with_structured_output.invoke([
        SystemMessage(content="Analyze the following search results and determine their relevance to the target information. Provide a score between 0 and 100 for each result based on its relevance. Return the results in the same format as the input, including the new 'score' field."), 
        HumanMessage(content= yaml.dump({
            
            "search_results": search_results
        }))
    ])
    print("response")
    print(response.model_dump_json())
    relevant_search_results = []

    response_dict = response.model_dump()

  
    for search_result in search_results:
        print(search_result["property_name"])
        # map back and assign the score to the search result
        if search_result["property_name"] in response_dict:
            search_result["score"] = response_dict[search_result["property_name"]]["relevance_score"]
            search_result["official_source"] = response_dict[search_result["property_name"]]["is_official_source_for_target_information"]
            search_result["relevant"] = response_dict[search_result["property_name"]]["is_relevant_to_target_information"]
            search_result["is_outdated"] = response_dict[search_result["property_name"]]["is_outdated"]
            
        else:
            print(f"No score found for {search_result['title']}")
            search_result["score"] = 0

    
    non_outdated_search_results = [search_result for search_result in search_results if not search_result["is_outdated"]]

    if state['include_unofficial_sources']:
        relevant_search_results = non_outdated_search_results
    else:
        relevant_search_results = [search_result for search_result in non_outdated_search_results if search_result["official_source"]]

    if state['only_relevant_sources']:
        relevant_search_results = [search_result for search_result in relevant_search_results if search_result["relevant"]]
    

    return {"relevant_search_results": relevant_search_results}

def get_next_urls_to_scrape(state: StructuredWebSearcherState, skip_urls: List[str]):
    search_results = state['relevant_search_results']
    scraping_batch_size = state['scraping_batch_size']
    
   
    # get the top p most relevant search results according to the score
    scrape_candidates = [result for result in search_results if result["link"] not in skip_urls]
    if state['prefer_official_sources']: #order by official sources first and then by score
        scrape_candidates = sorted(scrape_candidates, key=lambda x: (x['official_source'], x['score']), reverse=True)
    else:
        scrape_candidates = sorted(scrape_candidates, key=lambda x: x['score'], reverse=True)

    relevant_search_results = scrape_candidates[:scraping_batch_size]
    return [result["link"] for result in relevant_search_results]

async def scrape_search_results(state: StructuredWebSearcherState):
    print("🚩 scrape_search_results", f"scraping {len(state['relevant_search_results'])} of {len(state['search_results'])} search results")
    scraping_bee_service = ScrapingBeeService(max_concurrent_pages=5,verbose=True)
    skip_urls = []
    scraped_results = {}
    while True:
        print("scraping next batch")
        # stop scraping if we have enough results
        if len(scraped_results.keys()) >= len(state['relevant_search_results']) * state["top_p_search_results"]:
            break
        next_urls_to_scrape = get_next_urls_to_scrape(state, skip_urls)
        # stop scraping if we have no more urls to scrape
        if len(next_urls_to_scrape) == 0:
            break
        scraped_results.update(await scraping_bee_service.scrape_urls(
            next_urls_to_scrape, 
            max_cost_per_page=state['max_cost_per_scrape'],
            timeout=state['scraping_timeout']
        ))
        skip_urls.extend(next_urls_to_scrape)
        

    # Create screenshots directory if it doesn't exist
    screenshots_dir = os.path.join(os.getcwd(), "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    
    # Save screenshots
    screenshots = {}
    for url, contents in scraped_results.items():
        for content in contents:
            if content.content_type == "image/png":
                # Create a sanitized filename from the URL
                parsed_url = urlparse(url)
                sanitized_domain = re.sub(r'[^a-zA-Z0-9]', '_', parsed_url.netloc)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{sanitized_domain}_{timestamp}.png"
                filepath = os.path.join(screenshots_dir, filename)
                
                # Save the screenshot
                with open(filepath, 'wb') as f:
                    f.write(content.content)
                
                screenshots[url] = filepath
                print(f"📸 Saved screenshot: {filepath}")
                break
    
    formatted_search_results = format_sources(scraped_results, state['relevant_search_results'])

    return {
        "formatted_search_results": formatted_search_results,
        "screenshots": screenshots
    }

def summarize_content(state: StructuredWebSearcherState):
    print("🚩 summarize_content")
    formatted_search_results = state['formatted_search_results']
    prompt = load_prompt("content_summarizer_prompt.txt")
    

    response = state['model'].invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=[ 
            {
                "type": "text",
                "text": yaml.dump({
                    "target_information": state['target_information'],
                    "additional_instructions": state['additional_summarizer_instructions'],
                    "focus on this information": state['output_schema'].model_json_schema() if is_pydantic_model(state['output_schema']) else None,
                })
            }, 
            {
                "type": "text",
                "text": "sources:"
            },
            *formatted_search_results
        ])]
    )

    print("summarized content")
    print(response.content)
 
    return {"summarized_content": response.content}

def generate_output(state: StructuredWebSearcherState):
    structured_llm = state['model'].with_structured_output(state['output_schema'], method="json_schema")
    model_response = structured_llm.invoke([
        SystemMessage(content=load_prompt("output_generator_prompt.txt")),
        state['target_information_source']
    ])
    if is_pydantic_model(state['output_schema']):
        final_output = model_response.model_dump()
    else:
        final_output = model_response

    return {"output": final_output}

def prepare_target_information_source(state: StructuredWebSearcherState):
    print("🚩 prepare_target_information_source")
    if state['needs_summarization']:
        return {"target_information_source": HumanMessage(content=[
            {
                "type": "text",
                "text": yaml.dump({
                    "target_information": state['target_information'],
                    "additional_instructions": state['additional_formatting_instructions'],
                    "output_schema": state['output_schema'].model_json_schema() if is_pydantic_model(state['output_schema']) else None,
                    "summarized_content": state['summarized_content']
                })
            }
        ])}
    elif state["needs_scraping"]:
        print("🚩 prepare_target_information_source", "scraping")
        message = HumanMessage(content=[ 
            {
            "type": "text",
            "text": yaml.dump({
                    "target_information": state['target_information'],
                    "additional_instructions": state['additional_formatting_instructions'],
        
                })
            }, 
            *state["formatted_search_results"]
        ])
        return {"target_information_source": message}	
    else:
        return {"target_information_source": HumanMessage(content=[{
            "type": "text",
            "text": yaml.dump({
                "target_information": state['target_information'],
                "additional_instructions": state['additional_formatting_instructions'],
                "search_results": state['search_results']
            })
        }])}
   
    
def check_needs_query_generation(state: StructuredWebSearcherState):
    print("🚩 check_needs_query_generation")
    if not "queries" in state:
        return True
    return state['queries'] is None

def check_is_summarizing_necessary(state: StructuredWebSearcherState):
    print("🚩 check_is_summarizing_necessary")
    return state['needs_summarization']

def check_is_scraping_necessary(state: StructuredWebSearcherState):
    print("🚩 check_is_scraping_necessary")
    return state['needs_scraping']

def check_has_relevant_search_results(state: StructuredWebSearcherState):
    print("🚩 check_has_relevant_search_results")
    return len(state['relevant_search_results']) > 0

def check_has_accomplished_goal(state: StructuredWebSearcherState):
    print("🚩 check_has_accomplished_goal")
    return state['output'] is not None


def get_web_searcher_graph():
    web_searcher_graph = StateGraph(state_schema=StructuredWebSearcherState, input=StructuredWebResearcherInputSchema)

    web_searcher_graph.add_node("generate_queries", generate_queries)
    web_searcher_graph.add_node("execute_queries", execute_queries)
    web_searcher_graph.add_node("analyze_search_result_relevance", analyze_search_result_relevance)
    web_searcher_graph.add_node("scrape_search_results", scrape_search_results)
    web_searcher_graph.add_node("prepare_target_information_source", prepare_target_information_source)
    web_searcher_graph.add_node("summarize_content", summarize_content)
    web_searcher_graph.add_node("generate_output", generate_output)

    web_searcher_graph.add_conditional_edges(START, check_needs_query_generation, {True: "generate_queries", False: "execute_queries"})
    web_searcher_graph.add_edge("generate_queries", "execute_queries")

    web_searcher_graph.add_conditional_edges("execute_queries", check_is_scraping_necessary, {True: "analyze_search_result_relevance", False: "prepare_target_information_source"})

    web_searcher_graph.add_conditional_edges("analyze_search_result_relevance", check_has_relevant_search_results, {True: "scrape_search_results", False: "prepare_target_information_source"})
    web_searcher_graph.add_conditional_edges("scrape_search_results", check_is_summarizing_necessary, {True: "summarize_content", False: "prepare_target_information_source"})
    web_searcher_graph.add_edge("summarize_content", "prepare_target_information_source")


    web_searcher_graph.add_edge("prepare_target_information_source", "generate_output")
    web_searcher_graph.add_edge("generate_output", END)
    

    return web_searcher_graph.compile()


if __name__ == "__main__":
    pass
