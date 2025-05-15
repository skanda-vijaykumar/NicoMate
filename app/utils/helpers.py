import os
import logging
import re
from typing import Dict, List, Any, Optional, Tuple


def ensure_directory_exists(directory_path: str) -> bool:
    """Ensure directory exists, create if it doesn't."""
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logging.info(f"Created directory: {directory_path}")
        return True
    except Exception as e:
        logging.error(f"Error creating directory {directory_path}: {str(e)}")
        return False


def extract_connector_families(query_text: str) -> List[str]:
    """Extract mentioned connector families from text."""
    mentioned_families = []
    query_upper = query_text.upper()
    valid_families = ["AMM", "CMM", "DMM", "EMM", "DBM", "DFM"]

    for family in valid_families:
        if family in query_upper:
            mentioned_families.append(family)
    return mentioned_families


def normalize_awg_value(awg_value) -> Optional[int]:
    """Normalize AWG value to an integer."""
    if isinstance(awg_value, (int, float)):
        return int(awg_value)
    elif isinstance(awg_value, str):
        awg_str = awg_value.upper()
        if "AWG" in awg_str:
            try:
                return int(awg_str.replace("AWG", ""))
            except ValueError:
                pass
    # Return None if conversion failed
    return None


def clean_text_for_log(text: str, max_length: int = 100) -> str:
    """Clean text for logging, truncating if needed."""
    if text is None:
        return "None"

    # Replace newlines with spaces
    text = text.replace("\n", " ")

    # Truncate if too long
    if len(text) > max_length:
        return text[:max_length] + "..."

    return text
