from .schemas import QueryGeneratorOutputSchema, SearchResultRelevanceAnalysis, StructuredWebSearcherState
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from services.SerperSearchService import SerperSearchService
from services.ScrapingBeeService import ScrapingBeeService
import yaml
from utils.formatting_utils import format_sources
from utils.type_checking_utils import is_pydantic_model
import os
from urllib.parse import urlparse
from datetime import datetime
from .utils import load_prompt, get_next_urls_to_scrape
from pydantic import create_model
from typing import List
import re
from langchain_core.runnables import RunnableConfig


def generate_queries(state: StructuredWebSearcherState):
    print("🚩 generate_queries")

    print(state)

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

    return {"queries": queries}



def execute_queries(state: StructuredWebSearcherState):
    print("🚩 execute_queries")
    serper_search_service = SerperSearchService()
    search_results = serper_search_service.search_google([ {"q": query, "num": 10} for query in state.queries])
    return {"search_results": search_results}

def filter_search_results(state: StructuredWebSearcherState):
    print("🚩 analyze_search_results")
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
    print("model output")
    print(model_output.model_dump_json())


    response_dict = model_output.model_dump()

  
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
        return {"relevant_search_results": filtered_search_results, "retries": state.retries, "failed_queries": state.failed_queries, "queries": state.queries}
 

    return {"relevant_search_results": filtered_search_results}



async def scrape_search_results(state: StructuredWebSearcherState):
    print("🚩 scrape_search_results", f"scraping {len(state.relevant_search_results)} of {len(state.search_results)} search results")
    scraping_bee_service = ScrapingBeeService(max_concurrent_pages=5,verbose=True)
    skip_urls = []
    scraped_results = {}
    while True:
        print("scraping next batch")
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
            timeout=state.scraping_timeout
        ))
        skip_urls.extend(next_urls_to_scrape)

    
    if len(scraped_results.keys()) == 0:
        state.all_scrapes_failed = True
        state.retries += 1
        state.failed_queries.extend(state.queries)
        state.queries = None # reset queries
        return {"all_scrapes_failed": state.all_scrapes_failed, "retries": state.retries, "failed_queries": state.failed_queries, "queries": state.queries}

  
    
    formatted_search_results = format_sources(scraped_results, state.relevant_search_results)

    return {
        "formatted_search_results": formatted_search_results,
    }

def summarize_content(state: StructuredWebSearcherState):
    print("🚩 summarize_content")
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
                    "focus on this information": state.output_schema.model_json_schema() if is_pydantic_model(state.output_schema) else None,
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
    structured_llm = state.model.with_structured_output(state.output_schema, method="json_schema")
    model_response = structured_llm.invoke([
        SystemMessage(content=load_prompt("output_generator_prompt.txt")),
        state.target_information_source
    ])
    if is_pydantic_model(state.output_schema):
        final_output = model_response.model_dump()
    else:
        final_output = model_response

    return {"output": final_output}

def prepare_target_information_source(state: StructuredWebSearcherState):
    print("🚩 prepare_target_information_source")
    if state.needs_summarization:
        return {"target_information_source": HumanMessage(content=[
            {
                "type": "text",
                "text": yaml.dump({
                    "target_information": state.target_information,
                    "additional_instructions": state.additional_formatting_instructions,
                    "output_schema": state.output_schema.model_json_schema() if is_pydantic_model(state.output_schema) else None,
                    "summarized_content": state.summarized_content
                })
            }
        ])}
    elif state.needs_scraping:
        print("🚩 prepare_target_information_source", "scraping")
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
    print("🚩 check_starting_point")
    if state.retries >= state.max_retries:
        return "max_retries_reached"
    
    if not hasattr(state, "queries") or state.queries is None:
        return "needs_queries"
    
    return "has_queries"

def check_is_summarizing_necessary(state: StructuredWebSearcherState):
    print("🚩 check_is_summarizing_necessary")
    if state.all_scrapes_failed:
        return "all_scrapes_failed"
    return "needs_summarization" if state.needs_summarization else "skip_summarization"

def check_is_scraping_necessary(state: StructuredWebSearcherState):
    print("🚩 check_is_scraping_necessary")
    return "requires_scraping" if state.needs_scraping else "skip_scraping"

def check_has_relevant_search_results(state: StructuredWebSearcherState):
    print("🚩 check_has_relevant_search_results")
    return "has_results" if len(state.relevant_search_results) > 0 else "no_results"

def check_has_accomplished_goal(state: StructuredWebSearcherState):
    print("🚩 check_has_accomplished_goal")
    return state.output is not None

def prepare_run(state: StructuredWebSearcherState):
    print("🚩 prepare_run")
    return {}



def get_web_searcher_graph():
    web_searcher_graph = StateGraph(state_schema=StructuredWebSearcherState)

    
    web_searcher_graph.add_node("prepare_run", prepare_run) # helper node

    web_searcher_graph.add_node("generate_queries", generate_queries)
    web_searcher_graph.add_node("execute_queries", execute_queries)
    
    web_searcher_graph.add_node("filter_search_results", filter_search_results)
    web_searcher_graph.add_node("scrape_search_results", scrape_search_results)
    web_searcher_graph.add_node("prepare_target_information_source", prepare_target_information_source)
    web_searcher_graph.add_node("summarize_content", summarize_content)
    web_searcher_graph.add_node("generate_output", generate_output)

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
        {"has_results": "scrape_search_results", "no_results": "prepare_run"}
    )


    
    web_searcher_graph.add_conditional_edges(
        "scrape_search_results", 
        check_is_summarizing_necessary, 
        {"needs_summarization": "summarize_content", "skip_summarization": "prepare_target_information_source", "all_scrapes_failed": "prepare_run"}
    )

    web_searcher_graph.add_edge("summarize_content", "prepare_target_information_source")


    web_searcher_graph.add_edge("prepare_target_information_source", "generate_output")
    web_searcher_graph.add_edge("generate_output", END)
    

    return web_searcher_graph.compile()


