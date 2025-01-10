from .schemas import QueryGeneratorOutputSchema, SearchResultRelevanceAnalysis, StructuredWebSearcherState
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from services.SerperSearchService import SerperSearchService
from services.ScrapingBeeService import ScrapingBeeService
import yaml
from utils.formatting_utils import format_sources
from utils.type_checking_utils import is_pydantic_model
from utils.logging import _log
from .utils import load_prompt, get_next_urls_to_scrape
from pydantic import create_model
import re
import json



def generate_queries(state: StructuredWebSearcherState):
    """Node that generates search queries based on the target information.
    
    Args:
        state: Current state containing target information and query parameters
        
    Returns:
        dict: Contains 'queries' field with list of generated search queries
    """
    _log("🚩 generate_queries", verbose=state.verbose)
    _log(state.model_dump(), verbose=state.verbose)

    query_generator_prompt = load_prompt("query_generator_prompt.txt")
    structured_model = state.model.with_structured_output(QueryGeneratorOutputSchema, method="json_schema")

    response = structured_model.invoke([
        SystemMessage(content=query_generator_prompt),
        HumanMessage(content=yaml.dump({
            "user_input": state.target_information,
            "num_queries": state.num_queries,
            "additional_instructions": state.additional_query_generator_instructions if state.additional_query_generator_instructions else None,
            "failed_queries": state.failed_queries
        }))
    ])
  
    queries = [query.query for query in response.selected_and_refined_queries]
    _log("model output:", verbose=state.verbose)
    _log(response.model_dump_json(indent=2), verbose=state.verbose)

    return {"queries": queries}



def execute_queries(state: StructuredWebSearcherState):
    """Node that executes the generated search queries using search service.
    
    Args:
        state: Current state containing queries to execute
        
    Returns:
        dict: Contains 'search_results' field with raw search results
    """
    _log("🚩 execute_queries", verbose=state.verbose)
    _log(state.model_dump(), verbose=state.verbose)
    serper_search_service = SerperSearchService()
    search_results = serper_search_service.search_google([ {"q": query, "num": state.num_results_per_query} for query in state.queries])
    return {"search_results": search_results}

def filter_search_results(state: StructuredWebSearcherState):
    """Node that analyzes and filters search results based on relevance and other criteria.
    
    Args:
        state: Current state containing search results and filtering preferences
        
    Returns:
        dict: Contains 'relevant_search_results' and potentially retry-related fields if no results found
    """
    _log("🚩 analyze_search_results", verbose=state.verbose)
    _log(state.model_dump(), verbose=state.verbose)
    search_results = state.search_results
    structured_output_dict = {}
    for search_result in search_results:
        property_name = re.sub(r'[^a-zA-Z0-9_]', '_', search_result['title'])
        search_result["property_name"] = property_name
        structured_output_dict[property_name] = (SearchResultRelevanceAnalysis, ...)
    dynamic_model = create_model("SearchResultsRelevanceAnalysis", **structured_output_dict)
    llm_with_structured_output = state.model.with_structured_output(dynamic_model, method="json_schema")
    model_output = llm_with_structured_output.invoke([
        SystemMessage(content="Analyze the following search results and determine their relevance to the target information. Provide a score between 0 and 100 for each result based on its relevance. Return the results in the same format as the input, including the new 'score' field."), 
        HumanMessage(content= yaml.dump({
            
            "search_results": search_results
        }))
    ])
    _log("model output:", verbose=state.verbose)
    _log(model_output.model_dump_json(indent=2), verbose=state.verbose)


    response_dict = model_output.model_dump()

  
    for search_result in search_results:
        # map back and assign the score to the search result
        if search_result["property_name"] in response_dict:
            search_result["score"] = response_dict[search_result["property_name"]]["relevance_score"]
            search_result["official_source"] = response_dict[search_result["property_name"]]["is_official_source_for_target_information"]
            search_result["relevant"] = response_dict[search_result["property_name"]]["is_relevant_to_target_information"]
            search_result["is_outdated"] = response_dict[search_result["property_name"]]["is_outdated"]
            
        else:
            print(f"No score found for {search_result['title']}")
            search_result["score"] = 0

    # skip outdated search results

    if not state.include_outdated_sources:
        filtered_search_results = [search_result for search_result in search_results if not search_result["is_outdated"]]
    else:
        filtered_search_results = search_results

    if state.include_unofficial_sources:
        filtered_search_results = filtered_search_results
    else:
        filtered_search_results = [search_result for search_result in filtered_search_results if search_result["official_source"]]

    if state.only_relevant_sources:
        filtered_search_results = [search_result for search_result in filtered_search_results if search_result["relevant"]]

    if len(filtered_search_results) == 0:
        state.retries += 1
        state.failed_queries.extend(state.queries)
        state.queries = None # reset queries
        state.errors.append("No relevant search results found")
        _log("No relevant search results found", verbose=state.verbose)
        return {"relevant_search_results": filtered_search_results, "retries": state.retries, "failed_queries": state.failed_queries, "queries": state.queries}
 

    return {"relevant_search_results": filtered_search_results}



async def scrape_search_results(state: StructuredWebSearcherState):
    """Node that scrapes content from filtered search results.
    
    Args:
        state: Current state containing relevant search results to scrape
        
    Returns:
        dict: Contains 'formatted_search_results' or retry-related fields if scraping fails
    """
    _log("🚩 scrape_search_results", verbose=state.verbose)
    _log(f"scraping {len(state.relevant_search_results)} of {len(state.search_results)} search results", verbose=state.verbose)
    _log(state.model_dump(), verbose=state.verbose)
    scraping_bee_service = ScrapingBeeService(max_concurrent_pages=state.scrapingbee_concurrency_limit,verbose=state.verbose)
    skip_urls = []
    scraped_results = {}
    while True:
        _log("scraping next batch", verbose=state.verbose)
        # stop scraping if we have enough results
        if len(scraped_results.keys()) >= len(state.relevant_search_results) * state.top_p_search_results:
            break
        next_urls_to_scrape = get_next_urls_to_scrape(state, skip_urls)
        # stop scraping if we have no more urls to scrape
        if len(next_urls_to_scrape) == 0:
            break
        scraped_results.update(await scraping_bee_service.scrape_urls(
            next_urls_to_scrape, 
            max_cost_per_page=state.max_cost_per_scrape,
            timeout=state.scraping_timeout,
            return_screenshot=state.return_scraped_screenshots

        ))
        skip_urls.extend(next_urls_to_scrape)

    
    if len(scraped_results.keys()) == 0:
        state.all_scrapes_failed = True
        state.retries += 1
        state.failed_queries.extend(state.queries)
        state.queries = None # reset queries
        state.errors.append("Scraping failed for all search results")
        _log("Scraping failed for all search results", verbose=state.verbose)
        return {"all_scrapes_failed": state.all_scrapes_failed, "retries": state.retries, "failed_queries": state.failed_queries, "queries": state.queries}

  
    
    formatted_search_results = format_sources(scraped_results, state.relevant_search_results, max_chars_per_source=state.max_chars_per_scraped_source, max_images_per_source=state.max_images_per_scraped_source, verbose=state.verbose)

    return {
        "formatted_search_results": formatted_search_results,
    }

def summarize_content(state: StructuredWebSearcherState):
    """Node that summarizes the scraped content.
    
    Args:
        state: Current state containing formatted search results
        
    Returns:
        dict: Contains 'summarized_content' field with the summary
    """
    _log("🚩 summarize_content", verbose=state.verbose)
    _log(state.model_dump(), verbose=state.verbose)
    formatted_search_results = state.formatted_search_results
    prompt = load_prompt("content_summarizer_prompt.txt")
    

    response = state.model.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=[ 
            {
                "type": "text",
                "text": yaml.dump({
                    "target_information": state.target_information,
                    "additional_instructions": state.additional_summarizer_instructions,
                    "focus on this information": state.output_schema.model_json_schema() if is_pydantic_model(state.output_schema) else state.output_schema,
                })
            }, 
            {
                "type": "text",
                "text": "sources:"
            },
            *formatted_search_results
        ])]
    )

    _log("model output:", verbose=state.verbose)
    _log(json.dumps(response.content, indent=2) if isinstance(response.content, dict) else response.content, verbose=state.verbose)
 
    return {"summarized_content": response.content}

def generate_structured_output(state: StructuredWebSearcherState):
    """Node that generates structured output based on the processed information.
    
    Args:
        state: Current state containing target information source
        
    Returns:
        dict: Contains 'output' field with the final structured output
    """
    _log("🚩 generate_structured_output", verbose=state.verbose)
    _log(state.model_dump(), verbose=state.verbose)
    structured_llm = state.model.with_structured_output(state.output_schema, method="json_schema")
    model_response = structured_llm.invoke([
        SystemMessage(content=load_prompt("output_generator_prompt.txt")),
        state.target_information_source
    ])
    if is_pydantic_model(state.output_schema):
        final_output = model_response.model_dump()
    else:
        final_output = model_response

    _log("model output:", verbose=state.verbose)
    _log(json.dumps(final_output, indent=2) if isinstance(final_output, dict) else final_output, verbose=state.verbose)
        

    return {"output": final_output}

def generate_unstructured_output(state: StructuredWebSearcherState):
    """Node that generates unstructured output based on the processed information.
    
    Args:
        state: Current state containing processed information
        
    Returns:
        dict: Contains 'output' field with the unstructured output
    """
    _log("🚩 generate_unstructured_output", verbose=state.verbose)
    _log(state.model_dump(), verbose=state.verbose)
    final_output = state.model.invoke(
            [
                SystemMessage(content=load_prompt("output_generator_prompt.txt")),
                state.target_information_source 
            ]
        )
    _log("model output:", verbose=state.verbose)
    _log(json.dumps(final_output.content, indent=2) if isinstance(final_output.content, dict) else final_output.content, verbose=state.verbose)
    return {"output": final_output}

def prepare_target_information_source(state: StructuredWebSearcherState):
    """Node that prepares the information source for output generation.
    
    Args:
        state: Current state containing processed information
        
    Returns:
        dict: Contains 'target_information_source' field with formatted message
    """
    _log("🚩 prepare_target_information_source", verbose=state.verbose)
    _log(state.model_dump(), verbose=state.verbose)
    if state.summarize_scraped_contents and state.scrape_search_results:
        return {"target_information_source": HumanMessage(content=[
            {
                "type": "text",
                "text": yaml.dump({
                    "target_information": state.target_information,
                    "additional_instructions": state.additional_formatting_instructions,
                    "output_schema": state.output_schema.model_json_schema() if is_pydantic_model(state.output_schema) else state.output_schema,
                    "summarized_content": state.summarized_content
                })
            }
        ])}
    elif state.scrape_search_results:
        message = HumanMessage(content=[ 
            {
                "type": "text",
                "text": yaml.dump({
                    "target_information": state.target_information,
                    "additional_instructions": state.additional_formatting_instructions,
                })
            }, 
            *state.formatted_search_results
        ])
        return {"target_information_source": message}	
    else:
        return {"target_information_source": HumanMessage(content=[{
            "type": "text",
            "text": yaml.dump({
                "target_information": state.target_information,
                "additional_instructions": state.additional_formatting_instructions,
                "search_results": state.search_results
            })
        }])}
   
    
def check_starting_point(state: StructuredWebSearcherState):
    _log("🚩 check_starting_point", verbose=state.verbose)
    _log(state.model_dump(), verbose=state.verbose)
    if state.retries >= state.max_retries:
        return "max_retries_reached"
    
    if not hasattr(state, "queries") or state.queries is None:
        return "needs_queries"
    
    return "has_queries"

def check_is_summarizing_necessary(state: StructuredWebSearcherState):
    _log("🚩 check_is_summarizing_necessary", verbose=state.verbose)
    
    if state.all_scrapes_failed:
        _log("all_scrapes_failed", verbose=state.verbose)
        return "all_scrapes_failed"
    _log("summarize_scraped_contents" if state.summarize_scraped_contents else "skip_summarization", verbose=state.verbose)
    return "summarize_scraped_contents" if state.summarize_scraped_contents else "skip_summarization"

def check_is_scraping_necessary(state: StructuredWebSearcherState):
    _log("🚩 check_is_scraping_necessary", verbose=state.verbose)
    _log("scrape" if state.scrape_search_results else "skip_scraping", verbose=state.verbose)
    return "requires_scraping" if state.scrape_search_results else "skip_scraping"

def check_has_relevant_search_results(state: StructuredWebSearcherState):
    _log("🚩 check_has_relevant_search_results", verbose=state.verbose)

    return "has_results" if len(state.relevant_search_results) > 0 else "no_results"



def prepare_run(state: StructuredWebSearcherState):
    _log("🚩 prepare_run", verbose=state.verbose)
    _log(state.model_dump(), verbose=state.verbose)
    # reset all_scrapes_failed flag
    return {"all_scrapes_failed": False}

def check_is_formatting_necessary(state: StructuredWebSearcherState):
    _log("🚩 check_is_formatting_necessary", verbose=state.verbose)
    _log("needs_formatting" if state.needs_formatting else "skip_formatting", verbose=state.verbose)
    return "needs_formatting" if state.needs_formatting else "skip_formatting"


def get_web_searcher_graph():
    """Creates and returns a compiled state graph for structured web searching.
    
    Returns:
        Compiled StateGraph: Graph defining the web searching workflow
    """
    web_searcher_graph = StateGraph(state_schema=StructuredWebSearcherState)

    
    web_searcher_graph.add_node("prepare_run", prepare_run) # helper node

    web_searcher_graph.add_node("generate_queries", generate_queries)
    web_searcher_graph.add_node("execute_queries", execute_queries)
    
    web_searcher_graph.add_node("filter_search_results", filter_search_results)
    web_searcher_graph.add_node("scrape", scrape_search_results)
    web_searcher_graph.add_node("prepare_target_information_source", prepare_target_information_source)
    web_searcher_graph.add_node("summarize_content", summarize_content)
    web_searcher_graph.add_node("generate_structured_output", generate_structured_output)
    web_searcher_graph.add_node("generate_unstructured_output", generate_unstructured_output)

    web_searcher_graph.add_edge( START, "prepare_run")

    web_searcher_graph.add_conditional_edges(
        "prepare_run", 
        check_starting_point, 
        {"needs_queries": "generate_queries", "has_queries": "execute_queries", "max_retries_reached": END}
    )
    web_searcher_graph.add_edge("generate_queries", "execute_queries")

    web_searcher_graph.add_conditional_edges(
        "execute_queries", 
        check_is_scraping_necessary, 
        {"requires_scraping": "filter_search_results", "skip_scraping": "prepare_target_information_source"}
    )

    web_searcher_graph.add_conditional_edges(
        "filter_search_results", 
        check_has_relevant_search_results, 
        {"has_results": "scrape", "no_results": "prepare_run"}
    )


    
    web_searcher_graph.add_conditional_edges(
        "scrape", 
        check_is_summarizing_necessary, 
        {"summarize_scraped_contents": "summarize_content", "skip_summarization": "prepare_target_information_source", "all_scrapes_failed": "prepare_run"}
    )

    web_searcher_graph.add_edge("summarize_content", "prepare_target_information_source")


    web_searcher_graph.add_conditional_edges(
        "prepare_target_information_source", 
        check_is_formatting_necessary, 
        {"needs_formatting": "generate_structured_output", "skip_formatting": "generate_unstructured_output"}
    )
    web_searcher_graph.add_edge("generate_structured_output", END)
    web_searcher_graph.add_edge("generate_unstructured_output", END)
    

    return web_searcher_graph.compile()


