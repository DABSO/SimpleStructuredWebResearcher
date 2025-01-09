from StructuredWebResearcher.StructuredWebResearcher import structured_web_searcher
from pydantic import BaseModel
from typing import Dict, Any, List
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio

load_dotenv()



class WebUrl(BaseModel):
        url: str


result = asyncio.run(structured_web_searcher.ainvoke({
        "target_information": "LinkedIn profile of Paperpal", 
        "output_schema": WebUrl, 
        "model": ChatOpenAI(model="gpt-4o-mini", temperature=0),
        "needs_scraping": False,
        "needs_summarization": False,
        "num_queries": 1,
        
        "additional_formatting_instructions": "Format the output as a JSON object with a single key 'url' and the value as the URL.",
        "additional_query_generator_instructions": "Generate a query to search for the LinkedIn profile of Paperpal.",
        "additional_summarizer_instructions": "Summarize the content of the LinkedIn profile of Paperpal.",
        }))
print(result["output"])