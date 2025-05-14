import pytest
import asyncio
from app.core.connector import LLMConnectorSelector

@pytest.fixture
def connector_selector():
    return LLMConnectorSelector()

def test_normalize_awg_value(connector_selector):
    assert connector_selector.normalize_awg_value(24) == 24
    assert connector_selector.normalize_awg_value("AWG24") == 24
    assert connector_selector.normalize_awg_value("invalid") is None

@pytest.mark.asyncio
async def test_process_initial_message(connector_selector):
    # Test processing a simple message
    message = "I need a connector with 2mm pitch and plastic housing"
    result = await connector_selector.process_initial_message(message)
    
    # Should be a dict with status
    assert isinstance(result, dict)
    assert "status" in result
    
    # Should extract pitch_size
    assert 'pitch_size' in connector_selector.answers
    pitch_value, confidence = connector_selector.answers['pitch_size']
    assert pitch_value == 2.0
    assert confidence > 0.5
    
    # Should extract housing_material
    assert 'housing_material' in connector_selector.answers
    material_value, confidence = connector_selector.answers['housing_material']
    assert material_value == "plastic"
    assert confidence > 0.5
    
    # Check that confidence scores were calculated
    assert connector_selector.confidence_scores["CMM"] > 0

@pytest.mark.asyncio
async def test_calculate_connector_score(connector_selector):
    # Test scoring with a simple answer set
    answers = {
        'pitch_size': (2.0, 0.9),
        'housing_material': ('plastic', 0.9),
        'mixed_power_signal': (False, 0.8)
    }
    
    # Calculate scores for different connectors
    cmm_score = connector_selector.calculate_connector_score(
        connector_selector.connectors['CMM'], 
        answers
    )
    dmm_score = connector_selector.calculate_connector_score(
        connector_selector.connectors['DMM'], 
        answers
    )
    
    # CMM should score higher for this combination
    assert cmm_score > dmm_score
    
    # Add a contradictory requirement
    answers['housing_material'] = ('metal', 0.9)
    
    # Recalculate scores
    cmm_score_new = connector_selector.calculate_connector_score(
        connector_selector.connectors['CMM'], 
        answers
    )
    dmm_score_new = connector_selector.calculate_connector_score(
        connector_selector.connectors['DMM'], 
        answers
    )
    
    # Now DMM should score higher
    assert dmm_score_new > cmm_score_new
    # Original score for CMM should be higher than the new score
    assert cmm_score > cmm_score_new