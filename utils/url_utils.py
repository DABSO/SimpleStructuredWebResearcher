from typing import Dict

# Comprehensive mapping of TLDs to their associated country codes
TLD_COUNTRY_MAP: Dict[str, str] = {
    # Generic TLDs (no specific country association)
    'com': 'us',
    'org': 'us',
    'net': 'us',
    'edu': 'us',
    'gov': 'us',
    'int': 'us',
    'mil': 'us',
    
    # Modern generic TLDs
    'io': 'us',  # Technically British Indian Ocean Territory, but commonly used for tech
    'ai': 'us',  # Technically Anguilla, but commonly used for AI companies
    'dev': 'us',
    'app': 'us',
    'co': 'us',  # Often used as generic TLD
    'me': 'us',  # Technically Montenegro, but commonly used personally
    'tv': 'us',  # Technically Tuvalu, but used for media
    'tech': 'us',
    'cloud': 'us',
    'info': 'us',
    'biz': 'us',
    'xyz': 'us',
    
    # Special multi-part TLDs
    'co.uk': 'gb',
    'org.uk': 'gb',
    'me.uk': 'gb',
    'ltd.uk': 'gb',
    'plc.uk': 'gb',
    'net.uk': 'gb',
    'sch.uk': 'gb',
    'ac.uk': 'gb',
    'gov.uk': 'gb',
    'nhs.uk': 'gb',
    
    'co.jp': 'jp',
    'ne.jp': 'jp',
    'ac.jp': 'jp',
    'go.jp': 'jp',
    'or.jp': 'jp',
    
    'com.au': 'au',
    'net.au': 'au',
    'org.au': 'au',
    'edu.au': 'au',
    'gov.au': 'au',
    
    'co.nz': 'nz',
    'net.nz': 'nz',
    'org.nz': 'nz',
    
    'com.br': 'br',
    'net.br': 'br',
    'org.br': 'br',
    'gov.br': 'br',
    
    'com.mx': 'mx',
    'gob.mx': 'mx',
    'org.mx': 'mx',
    
    'co.in': 'in',
    'net.in': 'in',
    'org.in': 'in',
    'gov.in': 'in',
    
    'co.za': 'za',
    'org.za': 'za',
    'net.za': 'za',
    'gov.za': 'za',
    
    'com.sg': 'sg',
    'org.sg': 'sg',
    'gov.sg': 'sg',
    'edu.sg': 'sg',
    
    'com.cn': 'cn',
    'org.cn': 'cn',
    'net.cn': 'cn',
    'gov.cn': 'cn',
    
    'com.hk': 'hk',
    'org.hk': 'hk',
    'gov.hk': 'hk',
    'edu.hk': 'hk',
}

def get_country_code_from_url(url: str, default_country: str = 'us') -> str:
    """
    Extract country code from URL's TLD with comprehensive special cases handling.
    
    Args:
        url (str): The URL to analyze
        default_country (str): Default country code to return if no specific mapping exists
    
    Returns:
        str: Two-letter country code (ISO 3166-1 alpha-2)
    
    Examples:
        >>> get_country_code_from_url('https://example.com')
        'us'
        >>> get_country_code_from_url('https://example.co.uk')
        'gb'
        >>> get_country_code_from_url('https://example.de')
        'de'
        >>> get_country_code_from_url('https://example.ai')
        'us'
    """
    # Extract domain from URL
    try:
        domain = url.lower().split('//')[-1].split('/')[0]
    except (AttributeError, IndexError):
        return default_country
    
    # Remove any port numbers if present
    domain = domain.split(':')[0]
    
    # Handle special multi-part TLDs first
    domain_parts = domain.split('.')
    if len(domain_parts) >= 2:
        # Try two-part TLD (e.g., co.uk)
        if len(domain_parts) >= 3:
            two_part_tld = '.'.join(domain_parts[-2:])
            if two_part_tld in TLD_COUNTRY_MAP:
                return TLD_COUNTRY_MAP[two_part_tld]
        
        # Get the simple TLD
        tld = domain_parts[-1]
        
        # If it's a two-letter TLD and not in our special cases,
        # assume it's a country code
        if len(tld) == 2 and tld not in TLD_COUNTRY_MAP:
            return tld
        
        # Return mapped country code or default
        return TLD_COUNTRY_MAP.get(tld, default_country)
    
    return default_country