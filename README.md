# Structured Web Researcher

A simple, configurable web research agent built with LangGraph that can extract structured information from unstructured sources from the web. It uses a combination of search APIs, web scraping, and LLMs to gather and process information based on user queries. The agent is designed to function as a reusable module that can be integrated into larger graph-based workflows.

## Features

- Configurable search and scraping parameters
- Basic error handling and retries
- Structured output enforcement using Pydantic models
- Supports screenshots and text content extraction
- Filters results by relevance and source type

## Installation

1. Clone the repository
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Copy .env.example to .env and add your API keys:
```
   OPENAI_API_KEY="your-key"
   SCRAPINGBEE_API_KEY="your-key" 
   SERPER_API_KEY="your-key"
```
## Usage Example

Here's a simple example that searches for pricing plans:

```python
from StructuredWebResearcher import structured_web_searcher
from pydantic import BaseModel
from typing import List
from langchain_openai import ChatOpenAI

class PricingPlan(BaseModel):
    plan_name: str
    plan_price: str
    plan_features: List[str]

class PricingPlanList(BaseModel):
    plans: List[PricingPlan]

config = {
    "target_information": "Current yearly plans of lovable dev",
    "output_schema": PricingPlanList,
    "model": ChatOpenAI(model="gpt-4", temperature=0),
    "needs_scraping": True,
    "needs_summarization": True,
    "prefer_official_sources": True
}

result = await structured_web_searcher.ainvoke(config)
print(result["errors"])
print(result["output"])
```

## How it Works

The core functionality is implemented in two main files:

### schemas.py
Defines the state management and data structures using Pydantic models. The main class `StructuredWebSearcherState` handles:

- Query generation and tracking
- Search result filtering
- Scraping configuration
- Content processing settings

See [schemas.py](StructuredWebResearcher/schemas.py) for implementation details.

### graph.py 
Implements the LangGraph workflow with nodes for:

1. Query generation
2. Search execution
3. Result filtering
4. Web scraping
5. Content summarization
6. Output formatting

See [graph.py](StructuredWebResearcher/graph.py) for the full implementation.

## Limitations

- Relies on external APIs that may have rate limits/costs
- Basic error handling - may need enhancement for production use
- Limited to text and screenshot content types
- No caching implementation
- Simple retry mechanism

