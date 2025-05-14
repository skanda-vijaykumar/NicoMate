
import pytest
from app.utils.helpers import normalize_awg_value, extract_connector_families

def test_normalize_awg_value():
    # Test with integer
    assert normalize_awg_value(24) == 24
    
    # Test with string containing AWG
    assert normalize_awg_value("AWG24") == 24
    
    # Test with invalid value
    assert normalize_awg_value("invalid") is None

def test_extract_connector_families():
    # Test with multiple families
    query = "I need information about AMM and DMM connectors"
    families = extract_connector_families(query)
    assert "AMM" in families
    assert "DMM" in families
    assert len(families) == 2
    
    # Test with lowercase
    query = "What is the pitch size of amm connectors?"
    families = extract_connector_families(query)
    assert "AMM" in families
    assert len(families) == 1
    
    # Test with no families
    query = "Tell me about connectors"
    families = extract_connector_families(query)
    assert len(families) == 0