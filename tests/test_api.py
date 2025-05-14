import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

@pytest.mark.asyncio
async def test_new_session():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/new_session")
        assert response.status_code == 200
        json_data = response.json()
        
        assert "sessionId" in json_data
        assert "chatId" in json_data
        
        # Store session ID for later tests
        session_id = json_data["sessionId"]
        
        # Test getting the session
        response = await client.get(f"/session/{session_id}")
        assert response.status_code == 200
        session_data = response.json()
        
        assert "messages" in session_data
        # Should have at least the welcome message
        assert len(session_data["messages"]) >= 1

@pytest.mark.asyncio
async def test_chat_endpoint():
    # First create a session
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/new_session")
        session_id = response.json()["sessionId"]
        
        # Now test the chat endpoint
        response = await client.post(
            "/chat", 
            json={
                "sessionId": session_id,
                "message": "What is the pitch size of CMM connectors?"
            }
        )
        
        # We should get a streaming response
        assert response.status_code == 200
        
        # The response should be text/plain
        assert response.headers["content-type"] == "text/plain"

@pytest.mark.asyncio
async def test_invalid_session():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Try to get an invalid session
        response = await client.get("/session/invalid_session_id")
        assert response.status_code == 404
        
        # Try to chat with an invalid session
        response = await client.post(
            "/chat", 
            json={
                "sessionId": "invalid_session_id",
                "message": "Hello"
            }
        )
        assert response.status_code == 404