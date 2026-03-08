"""Query parsing utilities: brand extraction and battery-related validation."""

import re


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric/hyphen terms."""
    return re.findall(r"[a-z0-9-]+", text.lower())


def extract_brands_from_dataset(dataset):
    """Dynamically extract all unique brands from the dataset."""
    if not dataset:
        return []
    return sorted(set(item["brand"].lower() for item in dataset if "brand" in item))


def extract_domain_terms_from_dataset(dataset):
    """Build a domain vocabulary from dataset values (brand/device/chemistry/text)."""
    if not dataset:
        return set()

    terms = set()
    excluded_terms = {"devices", "device", "use", "uses", "general"}
    for item in dataset:
        for key in ("brand", "device", "chemistry", "text"):
            value = item.get(key, "")
            if value:
                terms.update(token for token in _tokenize(str(value)) if token not in excluded_terms)
    return terms


def extract_brand(query, dataset=None):
    """Extract brand name from query using regex patterns.
    
    Args:
        query: User query string
        dataset: Dataset to extract brands from (optional)
    
    Returns:
        Brand name if found, None otherwise
    """
    if not dataset:
        return None
        
    query_lower = query.lower()
    
    # Match brands using word boundaries for precision
    for item in dataset:
        brand = item["brand"].lower()
        if re.search(rf"\b{brand}\b", query_lower):
            return item["brand"]  # Return original case
    
    return None


def is_battery_related(query, dataset=None):
    """Check if query is related to batteries. Returns True only for battery queries.
    
    Args:
        query: User query string
        dataset: Optional dataset to extract brand names
    
    Returns:
        True if query is battery-related, False otherwise
    """
    if not dataset:
        return False

    query_lower = query.lower()

    # Brand mention is the strongest signal.
    brands = extract_brands_from_dataset(dataset)
    for brand in brands:
        if re.search(rf"\b{re.escape(brand)}\b", query_lower):
            return True

    # Otherwise, require overlap with terms learned from the dataset itself.
    query_terms = set(_tokenize(query_lower))
    domain_terms = extract_domain_terms_from_dataset(dataset)
    return len(query_terms.intersection(domain_terms)) > 0