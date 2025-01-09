from bs4 import BeautifulSoup
def has_cloudflare_challenge( html_content: str, verbose: bool = False) -> bool:
    """Checks if a webpage's HTML content indicates a Cloudflare challenge."""
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        html_lower = html_content.lower()

        
        # Check for challenge keywords
        challenge_keywords = [
            "verify you are human",
            "cloudflare-challenge",
        ]
        for keyword in challenge_keywords:
            if keyword in html_lower:
                if verbose:
                    print(f"Detected Cloudflare keyword: {keyword}")
                return True

        # Check iframes
        for iframe in soup.find_all('iframe'):
            src = iframe.get('src')
            if src and isinstance(src, str) and "cloudflare" in src.lower():
                if verbose:
                    print(f"Detected Cloudflare iframe: {src}")
                return True

        if verbose:
            print("No Cloudflare challenge detected")
        return False

    except Exception as e:
        error_msg = f"Failed to check for Cloudflare challenge: {str(e)}"
        if verbose:
            print(error_msg)
        raise RuntimeError(error_msg) from e