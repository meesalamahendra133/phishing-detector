from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlparse

from src.phiusiil.dataset import FEATURE_COLUMNS


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _is_ip_address(host: str) -> int:
    if not host:
        return 0

    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        return 0


def _get_tld(host: str) -> str:
    if not host or "." not in host:
        return ""

    return host.rsplit(".", 1)[-1]


def _count_subdomains(host: str) -> int:
    if not host:
        return 0

    parts = host.split(".")

    if len(parts) <= 2:
        return 0

    return len(parts) - 2


def _obfuscated_char_count(url: str) -> int:
    """
    Count percent-encoded characters such as %20, %2F, etc.
    """
    return len(re.findall(r"%[0-9A-Fa-f]{2}", url))


def _character_continuation_rate(url: str) -> float:
    """
    Measure the proportion of adjacent characters that belong
    to the same broad character class.

    Character classes:
        letter
        digit
        special
    """

    if len(url) <= 1:
        return 0.0

    def char_class(char: str) -> str:
        if char.isalpha():
            return "letter"

        if char.isdigit():
            return "digit"

        return "special"

    same_class = 0

    for first, second in zip(url, url[1:]):
        if char_class(first) == char_class(second):
            same_class += 1

    return same_class / (len(url) - 1)


def _url_character_probability(url: str) -> float:
    """
    Calculate the average empirical character probability
    within the URL.

    This is an inference-time approximation because the original
    PhiUSIIL dataset does not expose the exact training-time
    probability table used to construct URLCharProb.
    """

    if not url:
        return 0.0

    counts = Counter(url)
    length = len(url)

    probabilities = [
        count / length
        for count in counts.values()
    ]

    return sum(probabilities) / len(probabilities)


def extract_phiusiil_features(url: str) -> dict[str, float]:
    """
    Extract the URL-based features required by the project model.

    Returns:
        Dictionary containing the features expected by the
        XGBoost model.
    """

    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string.")

    url = url.strip()

    parsed = urlparse(url)

    host = parsed.hostname or ""
    path_and_query = f"{parsed.path}{parsed.params}{parsed.query}"

    url_length = len(url)
    domain_length = len(host)

    letters = sum(char.isalpha() for char in url)
    digits = sum(char.isdigit() for char in url)

    obfuscated_chars = _obfuscated_char_count(url)

    special_chars = sum(
    	1
    	for char in url
    	if not char.isalnum()
    )

    feature_values = {
        "URLLength": float(url_length),

        "DomainLength": float(domain_length),

        "IsDomainIP": float(
            _is_ip_address(host)
        ),

        "TLDLength": float(
            len(_get_tld(host))
        ),

        "NoOfSubDomain": float(
            _count_subdomains(host)
        ),

        "HasObfuscation": float(
            1 if obfuscated_chars > 0 else 0
        ),

        "NoOfObfuscatedChar": float(
            obfuscated_chars
        ),

        "ObfuscationRatio": float(
            _safe_ratio(obfuscated_chars, url_length)
        ),

        "NoOfLettersInURL": float(
            letters
        ),

        "LetterRatioInURL": float(
            _safe_ratio(letters, url_length)
        ),

        "NoOfDegitsInURL": float(
            digits
        ),

        "DegitRatioInURL": float(
            _safe_ratio(digits, url_length)
        ),

        "NoOfEqualsInURL": float(
            url.count("=")
        ),

        "NoOfQMarkInURL": float(
            url.count("?")
        ),

        "NoOfAmpersandInURL": float(
            url.count("&")
        ),

        "NoOfOtherSpecialCharsInURL": float(
            special_chars
        ),

        "SpacialCharRatioInURL": float(
            _safe_ratio(special_chars, url_length)
        ),

        "IsHTTPS": float(
            1 if parsed.scheme.lower() == "https" else 0
        ),

        "TLDLegitimateProb": 0.0,

        "URLCharProb": float(
            _url_character_probability(url)
        ),
    }

    return {
        feature: feature_values[feature]
        for feature in FEATURE_COLUMNS
    }
