"""
Common utility functions and helpers.
"""
from typing import List, Any
import hashlib
import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize text for processing.

    Args:
        text: Raw text string

    Returns:
        Normalized text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-]', '', text)
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    return text.strip()


def clean_concept_name(name: str) -> str:
    """
    Clean and normalize concept names for deduplication.

    Args:
        name: Raw concept name

    Returns:
        Cleaned concept name
    """
    # Convert to lowercase
    name = name.lower().strip()
    # Remove articles
    name = re.sub(r'\b(the|a|an)\b', '', name)
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name)
    # Remove special characters
    name = re.sub(r'[^\w\s]', '', name)
    return name.strip()


def generate_hash(text: str) -> str:
    """
    Generate SHA256 hash of text.

    Args:
        text: Text to hash

    Returns:
        Hex digest of hash
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0-1)
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def chunk_text_by_tokens(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separator: str = " "
) -> List[str]:
    """
    Split text into chunks by approximate token count.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in tokens (approximate)
        overlap: Overlap between chunks in tokens (approximate)
        separator: Token separator (default: space)

    Returns:
        List of text chunks
    """
    # Simple word-based chunking (approximating tokens)
    words = text.split(separator)
    chunks = []

    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(separator.join(chunk_words))
        i += chunk_size - overlap

    return chunks


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract top keywords from text using simple frequency analysis.

    Args:
        text: Input text
        top_n: Number of top keywords to return

    Returns:
        List of top keywords
    """
    # Lowercase and split
    words = text.lower().split()

    # Filter stopwords (simple list)
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their'
    }

    # Count word frequency
    word_freq = {}
    for word in words:
        word = re.sub(r'[^\w]', '', word)
        if word and word not in stopwords and len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Sort by frequency and return top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails

    Returns:
        Result of division or default
    """
    return numerator / denominator if denominator != 0 else default


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
