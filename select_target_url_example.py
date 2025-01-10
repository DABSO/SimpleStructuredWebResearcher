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
        "target_information": "LinkedIn profile of Paperpal ", 
        "queries": ["Paperpal linkedIn"],
        "output_schema": WebUrl, 
        "model": ChatOpenAI(model="gpt-4o-mini", temperature=0),
        "scrape_search_results": False,
        }))
print(result["output"])