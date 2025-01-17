from StructuredWebResearcher.StructuredWebResearcher import structured_web_searcher
from pydantic import BaseModel
from typing import Dict, Any, List
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio

load_dotenv()

class PricingPlan(BaseModel):
        plan_name: str
        plan_price: str
        plan_features: List[str]

class PricingPlanList(BaseModel):
    plans: List[PricingPlan]

config = {
    "target_information": "Current yearly plans of lovable dev", 
    "output_schema": PricingPlanList, 
    "model": ChatOpenAI(model="gpt-4o-mini", temperature=0, top_p=0.1),
    "scrape_search_results": True,
    "max_cost_per_scrape": 5,
    "summarize_scraped_contents": True,

    "top_p_search_results": 0.8,
    "prefer_official_sources": True,
    "only_relevant_sources": True,
    "include_unofficial_sources": True,
    "additional_summarizer_instructions": "Make sure to add the full list of features for each plan not just 'everything that is in plan xy'",
    "additional_formatting_instructions": "a free plan is also a plan if it is mentioned, add it to the list",
    "additional_query_generator_instructions": ""
}

result = asyncio.run(structured_web_searcher.ainvoke(config))
print(result["output"])