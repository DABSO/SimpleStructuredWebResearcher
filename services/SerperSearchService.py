import os

from typing import List, Optional, Dict, Any, Union
from itertools import groupby
import requests
from pydantic import BaseModel, Field


class SerperSearchService:
    """
    A class to interact with the Serper Search API for various types of searches.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the SerperSearch instance by loading the API key.

        Args:
            api_key (Optional[str]): The Serper API key. If not provided, it will be loaded from the environment variable 'SERPER_API_KEY'.
        
        Raises:
            ValueError: If the API key is not provided and not found in environment variables.
        """
        self.api_key = api_key or os.getenv('SERPER_API_KEY')
        if not self.api_key:
            raise ValueError("SERPER_API_KEY is not set in environment variables.")

        self.base_url = "https://google.serper.dev"

        self.headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }



    def search_google(self, queries: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        """
        Performs a standard Google search.

        Args:
            queries (Union[Dict[str, Any], List[Dict[str, Any]]]): A single query dict or a list of query dicts.

        Returns:
            Any: The JSON response from the API.
        """
        endpoint = "/search"
        return self._parse_results(self._post_request(endpoint, queries), "organic")

    def search_images(self, queries: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        """
        Performs an image search.

        Args:
            queries (Union[Dict[str, Any], List[Dict[str, Any]]]): A single query dict or a list of query dicts.

        Returns:
            Any: The JSON response from the API.
        """
        endpoint = "/images"
        return self._parse_results(self._post_request(endpoint, queries), "images")

    def search_videos(self, queries: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        """
        Performs a video search.

        Args:
            queries (Union[Dict[str, Any], List[Dict[str, Any]]]): A single query dict or a list of query dicts.

        Returns:
            Any: The JSON response from the API.
        """
        endpoint = "/videos"
        return self._parse_results(self._post_request(endpoint, queries), "videos")

    def search_places(self, queries: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        """
        Searches for places.

        Args:
            queries (Union[Dict[str, Any], List[Dict[str, Any]]]): A single query dict or a list of query dicts.

        Returns:
            Any: The JSON response from the API.
        """
        endpoint = "/places"
        return self._parse_results(self._post_request(endpoint, queries), "places")

    def search_maps(self, queries: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        """
        Searches Google Maps.

        Args:
            queries (Union[Dict[str, Any], List[Dict[str, Any]]]): A single query dict or a list of query dicts.

        Returns:
            Any: The JSON response from the API.
        """
        endpoint = "/maps"
        return self._parse_results(self._post_request(endpoint, queries), "places")

    def search_reviews(self, queries: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        """
        Searches for reviews. Requires either 'cid' or 'fid' in each query.

        Args:
            queries (Union[Dict[str, Any], List[Dict[str, Any]]]): A single query dict or a list of query dicts. Each dict must contain either 'cid' or 'fid'.

        Returns:
            Any: The JSON response from the API.
        
        Raises:
            ValueError: If 'cid' or 'fid' is missing in any query.
        """
        if isinstance(queries, list):
            payload = queries
            for query in payload:
                    if not (query.get("cid") or query.get("fid")):
                        raise ValueError("Each review query must contain either 'cid' or 'fid'.")
                
        else:
            if  not (queries.get("cid") or queries.get("fid")):
                raise ValueError("For reviews search, either 'cid' or 'fid' must be provided.")
        endpoint = "/reviews"
        return self._parse_results(self._post_request(endpoint, queries, is_reviews=True), "reviews")

    def search_news(self, queries: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        """
        Performs a news search and returns structured results.

        Args:
            queries (Union[NewsQuery, List[NewsQuery]]): A single query dict or a list of query dicts.

        Returns:
            NewsSearchResponse: Structured news search results.
        """
        endpoint = "/news"
        return self._parse_results(self._post_request(endpoint, queries), "news")
        


    def search_shopping(self, queries: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        """
        Performs a shopping search.

        Args:
            queries (Union[Dict[str, Any], List[Dict[str, Any]]]): A single query dict or a list of query dicts.

        Returns:
            Any: The JSON response from the API.
        """
        endpoint = "/shopping"
        return self._parse_results(self._post_request(endpoint, queries), "shopping")

    def search_lens(self, queries: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        """
        Performs an image reverse search using Google Lens.

        Args:
            queries (Union[Dict[str, Any], List[Dict[str, Any]]]): A single query dict or a list of query dicts. Each dict should contain the 'url' key for the image URL.

        Returns:
            Any: The JSON response from the API.
        """
        endpoint = "/lens"
        return self._post_request(endpoint, queries)

    def search_scholar(self, queries: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        """
        Searches Google Scholar.

        Args:
            queries (Union[Dict[str, Any], List[Dict[str, Any]]]): A single query dict or a list of query dicts.

        Returns:
            Any: The JSON response from the API.
        """
        endpoint = "/scholar"
        return self._post_request(endpoint, queries)

    def search_patents(self, queries: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        """
        Searches for patents.

        Args:
            queries (Union[Dict[str, Any], List[Dict[str, Any]]]): A single query dict or a list of query dicts.

        Returns:
            Any: The JSON response from the API.
        """
        endpoint = "/patents"
        return self._post_request(endpoint, queries)

    def _post_request(
        self, 
        endpoint: str, 
        queries: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> Any:
        """
        Internal method to send POST requests to the Serper API.

        Args:
            endpoint (str): The API endpoint.
            queries (Union[Dict[str, Any], List[Dict[str, Any]]]): The query or list of queries.
            is_reviews (bool): Flag indicating if the request is for reviews, which require additional validation.

        Returns:
            Any: The JSON response from the API.
        
        Raises:
            ValueError: If required parameters for reviews are missing.
        """
        
  

        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.post(url, headers=self.headers, json=queries)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e), "status_code": response.status_code if 'response' in locals() else None}

    def search(
        self, 
        queries: Union[Dict[str, Any], List[Dict[str, Any]]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        General search method that delegates to specific search type methods.

        Args:
            queries (Union[Dict[str, Any], List[Dict[str, Any]]]): A single query dict or a list of query dicts.
                Each query must include a 'type' field specifying the search type.
            **kwargs: Additional parameters to include in each query.

        Returns:
            Dict[str, Any]: Combined results from all search types, keyed by search type.
        
        Raises:
            ValueError: If any search type is unsupported or required parameters are missing.
            TypeError: If queries are not in the expected format.
        """
        type_method_map = {
            "search": self.search_google,
            "images": self.search_images,
            "videos": self.search_videos,
            "places": self.search_places,
            "maps": self.search_maps,
            "reviews": self.search_reviews,
            "news": self.search_news,
            "shopping": self.search_shopping,
            "lens": self.search_lens,
            "scholar": self.search_scholar,
            "patents": self.search_patents,
        }

        # Convert single query to list for uniform processing
        query_list = [queries] if isinstance(queries, dict) else queries
        if not isinstance(query_list, list):
            raise TypeError("queries must be a dict or a list of dicts")

        # Validate and prepare queries
        for query in query_list:
            if not isinstance(query, dict) or 'type' not in query:
                raise ValueError("Each query must be a dict with a 'type' field")
            query.update(kwargs)

        # Sort and group queries by type
        sorted_queries = sorted(query_list, key=lambda x: x['type'].lower())
        grouped_queries = {
            search_type: list(group)
            for search_type, group in groupby(sorted_queries, key=lambda x: x['type'].lower())
        }

        # Execute each group of queries with the appropriate method
        results = {}
        for search_type, type_queries in grouped_queries.items():
            method = type_method_map.get(search_type)
            if not method:
                raise ValueError(f"Unsupported search type: {search_type}")
            
            # Remove type field from queries before sending to method
            cleaned_queries = [{k: v for k, v in q.items() if k != 'type'} for q in type_queries]
            results[search_type] = method(cleaned_queries)

        

        return results

    def _parse_results(self, results: Dict[str, Any], key: str) -> Dict[str, Any]:
        if isinstance(results, dict) and key in results:
            return results[key]
        elif isinstance(results, list) and all(isinstance(item, dict) and key in item for item in results):
            target_pages = [item[key] for item in results]
            # flatten the list of lists
            return [item for sublist in target_pages for item in sublist]
        else:
            raise ValueError(f"Invalid response format for {key} search")
