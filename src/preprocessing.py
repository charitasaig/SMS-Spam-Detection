"""
Text Preprocessing Module for SMS Spam Message Detection.
Team 2 - Natural Language Processing

This module provides preprocessing functions designed specifically for SMS text:
- Preserves critical spam indicators: URLs, currency symbols, numeric codes, exclamation marks.
- Preserves semantic negations (e.g., 'not', 'no', 'never') to maintain message intent.
- Normalizes whitespace and case while handling missing/invalid input gracefully.
- Can be used both standalone and within a Scikit-learn Pipeline.
"""

import re
from typing import List, Optional

# Custom curated stop words list excluding negations and spam-sensitive tokens.
# Standard stop word lists often drop "not", "no", "never", which can invert meaning.
STANDARD_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
    "further", "had", "has", "have", "having", "he", "hed", "hes", "her", "here", "heres", "hers", 
    "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive", "if", "in", 
    "into", "is", "it", "its", "itself", "lets", "me", "more", "most", "my", "myself", "of", 
    "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", 
    "own", "same", "she", "shed", "shes", "should", "so", "some", "such", "than", "that", "thats", 
    "the", "their", "theirs", "them", "themselves", "then", "there", "theres", "these", "they", 
    "theyd", "theyll", "theyre", "theyve", "this", "those", "through", "to", "too", "under", 
    "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "what", "whats", "when", 
    "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why", "whys", "with", 
    "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves"
}

# Negation words that MUST be preserved for semantic clarity
NEGATION_WORDS = {"no", "not", "never", "none", "neither", "nor", "cannot", "without"}

# Stop words to filter (STANDARD_STOP_WORDS minus NEGATION_WORDS)
FILTER_STOP_WORDS = STANDARD_STOP_WORDS - NEGATION_WORDS

# Regex patterns for spam indicators
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:com|org|net|co|uk|biz|info|tv)\b\S*)", re.IGNORECASE)
CURRENCY_PATTERN = re.compile(r"[$£€¥₹]|(?:\b(?:gbp|eur|usd|inr|cash|prize)\b)", re.IGNORECASE)
PHONE_NUM_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b0800\d{4,}\b|\b\d{5,}\b")
EXCLAMATION_PATTERN = re.compile(r"!{1,}")
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALPHANUM_PATTERN = re.compile(r"[^a-z0-9\s]")


def clean_text(text: Optional[str], remove_stop_words: bool = True) -> str:
    """
    Cleans raw SMS message text while preserving critical spam indicators.
    
    Steps:
    1. Check for null or non-string input.
    2. Normalize URLs -> token 'url_token'
    3. Normalize Currency symbols -> token 'currency_token'
    4. Normalize Multi-digit codes / phone numbers -> token 'number_token'
    5. Tag excessive exclamation marks -> token 'exclamation_token'
    6. Lowercase conversion.
    7. Remove miscellaneous non-alphanumeric characters while preserving tokens.
    8. Tokenize and selectively remove non-negation stop words.
    9. Collapse whitespace and strip.
    
    Args:
        text: The raw input SMS text string.
        remove_stop_words: Whether to remove common non-negation stop words.
        
    Returns:
        Cleaned, normalized string ready for TF-IDF vectorization.
    """
    if text is None:
        return ""
    
    if not isinstance(text, str):
        text = str(text)
        
    text = text.strip()
    if not text:
        return ""

    # Replace URLs with recognizable token
    text = URL_PATTERN.sub(" url_token ", text)
    
    # Replace currency symbols/promotional currency words
    text = CURRENCY_PATTERN.sub(" currency_token ", text)
    
    # Replace long numbers, helpline codes, and phone numbers
    text = PHONE_NUM_PATTERN.sub(" number_token ", text)
    
    # Tag prominent exclamation marks (frequent in urgent spam like "WINNER!!!")
    text = EXCLAMATION_PATTERN.sub(" exclamation_token ", text)
    
    # Lowercase conversion
    text = text.lower()
    
    # Remove unwanted punctuation while preserving underscores in token names
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Tokenization & selective stop word removal
    tokens = text.split()
    
    if remove_stop_words:
        # Keep tokens if they are NOT in the filter list (or if they are negation tokens)
        tokens = [t for t in tokens if t not in FILTER_STOP_WORDS or t in NEGATION_WORDS]
        
    cleaned = " ".join(tokens)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def extract_message_features(raw_text: str) -> dict:
    """
    Extracts high-level descriptive metadata from an SMS message for UI inspection.
    
    Returns:
        dict containing char_length, word_count, has_url, has_currency, has_numbers,
        and exclamation_count.
    """
    if not isinstance(raw_text, str):
        raw_text = ""
        
    return {
        "char_length": len(raw_text),
        "word_count": len(raw_text.split()),
        "has_url": bool(URL_PATTERN.search(raw_text)),
        "has_currency": bool(CURRENCY_PATTERN.search(raw_text)),
        "has_numbers": bool(PHONE_NUM_PATTERN.search(raw_text)),
        "exclamation_count": len(EXCLAMATION_PATTERN.findall(raw_text))
    }
