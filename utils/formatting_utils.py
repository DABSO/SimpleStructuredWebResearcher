from services.ScrapingBeeService import ScrapedContent
from typing import Dict, List, Union, Any

import json
from typing import Union, Dict, Any

from bs4 import BeautifulSoup


from bs4 import NavigableString
from utils.conversion_utils import convert2message_parts

# TODO: Refactor this

def html_to_markdown(html_content):
    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Recursive function to process nodes
    def process_node(node, indent_level=0):
        markdown = []
        
        # Handle NavigableString (text nodes)
        if isinstance(node, NavigableString):
            text = str(node).strip()
            if text:
                markdown.append(text)
            return markdown

        if node.name in ['h1', 'h2', 'h3']:
            # Add headings with corresponding Markdown syntax
            level = '#' * int(node.name[1])
            markdown.append(f"{level} {node.get_text(strip=True)}\n")
        elif node.name == 'p':
            # Add paragraphs
            markdown.append(f"{node.get_text(strip=True)}\n")
        elif node.name == 'li':
            # Add list items with indentation for nested levels
            markdown.append(f"{'  ' * indent_level}- {node.get_text(strip=True)}")
        elif node.name in ['ul', 'ol']:
            # Recursively process child list items
            for child in node.find_all('li', recursive=False):
                markdown.extend(process_node(child, indent_level + 1))
        elif node.name is None:
            # Capture plain text nodes (e.g., text outside tags)
            text = node.strip()
            if text:
                markdown.append(text)
        
        # Process children for any other tags, in order
        for child in node.contents:
            markdown.extend(process_node(child, indent_level))
        
        return markdown

    # Process the entire HTML tree
    markdown_content = process_node(soup.body or soup)

    # Join and return the Markdown content
    return '\n'.join(markdown_content)

def format_sources(
    sources: Dict[str, List[ScrapedContent]],
    search_results: Union[Dict[str, Any], List[Dict[str, Any]]],
    max_chars_per_source: int = 5000,
    max_images_per_source: int = 5
) -> List[Dict[str, Any]]:
    """
    Returns a list of Message-Parts.
    """
    print("\n=== Starting format_sources ===")

    
    message_parts: List[Dict[str, Any]] = []

    # Process organic results
    for idx, item in enumerate(search_results):
        print(f"\n--- Processing organic result {idx + 1}/{len(search_results)} ---")
        
        url = item.get("link")
        if not url:
            print("-> Skipping: No URL found in result")
            continue

        message_parts.append({
            "type": "text",
            "text": "URL: " + url
        })

        print(f"-> Processing URL: {url}")
        scraped_contents = sources.get(url, [])
        print(f"-> Found {len(scraped_contents)} scraped content items")

        snippet = item.get("snippet")
        if snippet:
            message_parts.append({
                "type": "text",
                "text":  "Short Snippet: " + snippet
            })

        scraped_contents = sources.get(url, [])
        
        if not scraped_contents:
            print("-> Skipping: No scraped content found")
            message_parts.append({
                "type": "text",
                "text": "No scraped content found for URL " 
            })

            message_parts.append({
                "type": "text",
                "text": "\n-------\n" 
            })
            continue
            
        temp_list = []
        for sc_idx, sc in enumerate(scraped_contents):
            print(f"-> Converting scraped content {sc_idx + 1}/{len(scraped_contents)}")
            converted_parts = convert2message_parts(sc)
            if sc.content_type == "text/html":
                # parse with beautifulsoup and get the text 
                text_content = html_to_markdown(sc.content)
                converted_parts = [{"type": "text", "text":  text_content}]
            temp_list.extend(converted_parts)

        text_parts = [part for part in temp_list if part["type"] == "text"]
        non_text_parts = [part for part in temp_list if part["type"] != "text"]
        
        print(f"-> Conversion resulted in {len(text_parts)} text parts and {len(non_text_parts)} non-text parts")
        
        # Process text parts
        sum_chars = 0
        added_text_parts = 0
        while text_parts and sum_chars < max_chars_per_source:
            add_part = text_parts.pop(0)
            text_length = len(add_part["text"])
            
            if sum_chars + text_length > max_chars_per_source:
                print(f"-> Truncating text part to fit max_chars (current: {sum_chars}, adding: {text_length})")
                remaining_chars = max_chars_per_source - sum_chars
                add_part["text"] = add_part["text"][:remaining_chars]
                sum_chars = max_chars_per_source
            else:
                sum_chars += text_length
            
            message_parts.append(add_part)
            added_text_parts += 1
        
        # Add non-text parts
        added_non_text = min(len(non_text_parts), max_images_per_source)
        message_parts.extend(non_text_parts[:max_images_per_source] + [{"type": "text", "text": "\n-------\n"}])
        
        print(f"-> Added {added_text_parts} text parts ({sum_chars} chars) and {added_non_text} non-text parts")

    print(f"\n=== Finished format_sources: {len(message_parts)} total message parts ===\n")
    return message_parts
