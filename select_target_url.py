from StructuredWebResearcher.StructuredWebSearcher import get_web_searcher_graph
from pydantic import BaseModel
from typing import Dict, Any, List
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio

load_dotenv()

class StructuredWebResearcherInputSchema(BaseModel):
    target_information: str
    output_schema: BaseModel|Dict[str, Any]
    queries: List[str] | None = None
    needs_scraping: bool = True
    needs_summarization: bool = True

    num_queries: int = 1
    model : BaseChatModel
    relevance_threshold: int = 90

    additional_summarizer_instructions: str|None = None
    additional_query_generator_instructions: str|None = None
    additional_formatting_instructions: str|None = None


class WebUrl(BaseModel):
        url: str

graph = get_web_searcher_graph()
result = asyncio.run(graph.ainvoke({
        "target_information": "LinkedIn profile of Paperpal", 
        "output_schema": WebUrl, 
        "model": ChatOpenAI(model="gpt-4o-mini", temperature=0),
        "needs_scraping": False,
        "needs_summarization": False,
        "num_queries": 1,
        "relevance_threshold": 75,
        "additional_formatting_instructions": "Format the output as a JSON object with a single key 'url' and the value as the URL.",
        "additional_query_generator_instructions": "Generate a query to search for the LinkedIn profile of Paperpal.",
        "additional_summarizer_instructions": "Summarize the content of the LinkedIn profile of Paperpal.",
        }))
print(result["output"])