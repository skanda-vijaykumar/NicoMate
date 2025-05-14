import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_index_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # Check for key elements in the HTML
    assert "<title>Chatbot</title>" in response.text
    assert "chat-interface" in response.text
    assert "user-input" in response.text

@pytest.mark.asyncio
async def test_static_files():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Test CSS file is available
        response = await client.get("/static/styles.css")
        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]

@pytest.mark.asyncio
async def test_suggestion_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # First create a session
        response = await client.post("/new_session")
        session_id = response.json()["sessionId"]
        
        # Now test the suggestion endpoint
        response = await client.post(
            "/suggestion", 
            json={
                "sessionId": session_id,
                "message": "This is a system suggestion"
            }
        )
        
        assert response.status_code == 200
        assert response.json()["detail"] == "Suggestion submitted successfully."
        
        # Verify the suggestion was added to the session
        response = await client.get(f"/session/{session_id}")
        session_data = response.json()
        
        # Find the system message
        system_messages = [
            msg for msg in session_data["messages"] 
            if msg["sender"] == "system" and msg["content"] == "This is a system suggestion"
        ]
        
        assert len(system_messages) == 1