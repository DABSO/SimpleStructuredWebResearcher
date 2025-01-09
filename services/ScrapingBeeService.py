from scrapingbee import ScrapingBeeClient
import os
from typing import TypedDict, Optional, Union, Literal, List, Dict
from dataclasses import dataclass
import asyncio
from enum import Enum
from utils.webscraping_utils import has_cloudflare_challenge
from utils.url_utils import get_country_code_from_url

@dataclass
class ScrapedContent:
    content: Union[str, bytes]
    content_type: str
    encoding: Union[Literal["utf-8"], Literal["raw"]]

class ScrapingResult(TypedDict):
    screenshot: Optional[ScrapedContent]
    content: Optional[ScrapedContent]

class ProxyType(Enum):
    STANDARD = "standard"
    PREMIUM = "premium"
    STEALTH = "stealth"

class ScrapingBeeService:
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_concurrent_pages: int = 5, verbose: bool = False):
        # Only initialize once
        if not hasattr(self, 'initialized'):
            self.client = ScrapingBeeClient(api_key=os.getenv('SCRAPINGBEE_API_KEY'))
            self.semaphore = asyncio.Semaphore(max_concurrent_pages)
            self.active_semaphores = 0  # Add counter
            self.verbose = verbose
            self.initialized = True

    def _log(self, message: str) -> None:
        """Helper method to print messages only when verbose is True."""
        if self.verbose:
            print(message)



    async def scrape_url(self, url: str, return_screenshot: bool = True, return_content: bool = True, max_cost: int = 10, timeout: int = 10000, **scrapingbee_params):
        self._log(f"\n[SCRAPE] Starting scrape for: {url}")
        
        if not (return_screenshot or return_content):
            return None

        # Calculate how many attempts we can make with each proxy type
        stealth_attempts = max_cost // 75  # Most expensive but highest success rate
        remaining_cost = max_cost % 75
        
        premium_attempts = remaining_cost // 25
        remaining_cost = remaining_cost % 25
        
        standard_attempts = remaining_cost // 5

        self._log(f"[SCRAPE] Budget allocation: {stealth_attempts} stealth, {premium_attempts} premium, {standard_attempts} standard attempts")

        # try standard first (cheapest)
        for _ in range(standard_attempts):
            result = await self._scrape_with_scrapingbee(
                url=url, 
                return_screenshot=return_screenshot, 
                return_content=return_content, 
                proxy_type="standard"
            )
            if result:
                return result
            
        # Then premium
        for _ in range(premium_attempts):
            result = await self._scrape_with_scrapingbee(
                url=url, 
                return_screenshot=return_screenshot, 
                return_content=return_content, 
                proxy_type="premium"
            )
            if result:
                return result
            
        # try stealth last (highest success rate)
        for _ in range(stealth_attempts):
            result = await self._scrape_with_scrapingbee(
                url=url, 
                return_screenshot=return_screenshot, 
                return_content=return_content, 
                proxy_type="stealth",
                timeout=timeout,
                **scrapingbee_params
            )
            if result:
                return result

        

        

        self._log(f"[SCRAPE] ❌ All scraping attempts failed for: {url}")
        return None

  
        
  

    async def _scrape_with_scrapingbee(self, url: str, render: bool = True, 
                                      proxy_type: ProxyType = ProxyType.STANDARD,
                                      return_screenshot: bool = True, 
                                      return_content: bool = True, 
                                      timeout: int = 10000,
                                      **scrapingbee_params) -> Optional[List[ScrapedContent]]:
        self._log(f"[SCRAPINGBEE][{url}] Starting scrape with {proxy_type} proxy")
        params = {
            "render_js": render,
            "json_response": True,
            "wait_browser": "networkidle0",
            "block_ads": True,
            "block_resources": False,
        }
        
        params = {**params, **scrapingbee_params}
        # Use enum for proxy type
        if proxy_type == ProxyType.PREMIUM:
            params["premium_proxy"] = True
            params["country_code"] = get_country_code_from_url(url)
        elif proxy_type == ProxyType.STEALTH:
            params["stealth_proxy"] = True
            params["country_code"] = get_country_code_from_url(url)
        
        # Add screenshot parameter if needed
        if return_screenshot:
            params["screenshot_full_page"] = True
        

        
        
        try:
            self._log(f"[SCRAPINGBEE][{url}] Making request with params: {params}")
            response = await asyncio.to_thread(
                self.client.get, 
                url, 
                params=params,
            
            )
            self._log(f"[SCRAPINGBEE][{url}] Response type: {type(response)}")
            
            if  await self.check_if_blocked(response) :
                self._log(f"[SCRAPINGBEE][{url}] ❌ Response blocked with {proxy_type} proxy")
                return None

            self._log(f"[SCRAPINGBEE][{url}] ✅ Successfully received response with {proxy_type} proxy")
            # Handle the response based on content type
            response = response.json()
            if isinstance(response, dict):  # JSON response
                result: List[ScrapedContent] = []
                
                if return_screenshot and "screenshot" in response:
                    self._log(f"[SCRAPINGBEE][{url}] Processing screenshot...")
                    import base64
                    result.append(ScrapedContent(
                        content=base64.b64decode(response["screenshot"]),
                        content_type="image/png",
                        encoding="raw"
                    ))
                
                if return_content:
                    self._log(f"[SCRAPINGBEE][{url}] Processing content...")
                    if response["type"] == "html":
                        result.append(ScrapedContent(
                            content=response["body"],
                            content_type="text/html",
                            encoding="utf-8"
                        ))
                    elif response["type"] == "b64_bytes":  # If ScrapingBee target is PDFs, Images or other binary content
                        decoded_content = base64.b64decode(response["body"])
                        result.append(ScrapedContent(
                            content=decoded_content,
                            content_type=response.get("content_type", "application/octet-stream"),
                            encoding="raw"
                        ))
                
                return result if result else None
            
            self._log(f"[SCRAPINGBEE][{url}] ❌ Unexpected response format")
            return None

        except Exception as e:
            self._log(f"[SCRAPINGBEE][{url}] ❌ Request failed: {str(e)}")
            return None

  

    async def check_if_blocked(self, response) -> bool:
        """
        Determines if a response indicates the request was blocked.
        Expects a ScrapingBee response which returns JSON.
        """
        try:
            status_code = response.status_code
            self._log(f"[CHECK_BLOCKED] Response status code: {status_code}")

            # Parse JSON response
            try:
                json_response = response.json()
                content = json_response.get('body', '')
                self._log(f"[CHECK_BLOCKED] Successfully extracted HTML content from JSON, length: {len(content)}")
            except Exception as e:
                self._log(f"[CHECK_BLOCKED] Failed to parse JSON response: {str(e)}")
                return True  # If we can't parse the response, consider it blocked

            # Check status code
            if status_code in {403, 429} or status_code >= 400:
                self._log(f"[CHECK_BLOCKED] Blocked status code: {status_code}")
                return True

            # Check for blocking headers
            block_headers = ["x-captcha", "retry-after"]
            if any(header in response.headers for header in block_headers):
                self._log("[CHECK_BLOCKED] Blocking headers detected")
                return True

            # Check content length
            if not content or len(content) < 200:  # Adjust threshold as needed
                self._log(f"[CHECK_BLOCKED] Content too short: {len(content) if content else 0} chars")
                return True

            # Check for Cloudflare challenge
            if has_cloudflare_challenge(content, verbose=self.verbose):
                self._log("[CHECK_BLOCKED] Cloudflare challenge detected")
                return True

            self._log("[CHECK_BLOCKED] Response appears valid")
            return False

        except Exception as e:
            self._log(f"[CHECK_BLOCKED] Unexpected error: {str(e)}")
            return True


    async def scrape_urls(self, urls: List[str], return_screenshot: bool = True, return_content: bool = True, max_cost_per_page: int = 10, timeout: int = 10000) -> Dict[str, List[ScrapedContent]]:
        async def scrape_with_semaphore(url):
            self._log(f"[SEMAPHORE][{url}] Waiting to acquire... (Active: {self.active_semaphores}/{self.semaphore._value})")
            async with self.semaphore:
                self.active_semaphores += 1
                self._log(f"[SEMAPHORE][{url}] Acquired ✅ (Active: {self.active_semaphores}/{self.semaphore._value})")
                try:
                    result = await self.scrape_url(url, return_screenshot, return_content, max_cost_per_page, timeout=timeout)
                    return result
                finally:
                    self.active_semaphores -= 1
                    self._log(f"[SEMAPHORE][{url}] Released 🔓 (Active: {self.active_semaphores}/{self.semaphore._value})")
        
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build result dictionary, filtering out failed requests
        scraped_results = {}
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                self._log(f"[SCRAPE] Error scraping {url}: {result}")
                continue
            if result is None:
                self._log(f"[SCRAPE] No result for {url}")
                continue
            if result is not None:
                scraped_results[url] = result
                
        return scraped_results




