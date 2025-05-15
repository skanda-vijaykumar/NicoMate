import pytest
from httpx import AsyncClient, ASGITransport  
from fastapi.testclient import TestClient
from app.main import app
import os

# Set testing environment variables
os.environ["TESTING"] = "True"
os.environ["POSTGRES_DB"] = "alexis_test"  # Override with test database name

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

@pytest.fixture
async def mock_db():
    """Mock database setup for tests"""
    # Override database settings for testing
    os.environ["POSTGRES_DB"] = "alexis_test"
    
    # Initialize the database for testing
    from app.db.database import initialize_database
    try:
        initialize_database()
    except Exception as e:
        # Just log the error but continue, as sometimes DB is already initialized
        print(f"Database initialization warning (can be ignored if DB exists): {str(e)}")
    
    yield
    # No teardown needed for tests as the test database is destroyed after CI run

@pytest.fixture
async def initialized_app(mock_db):
    """Initialize app for testing"""
    from app.main import app_ready
    from app.api.dependencies import startup_complete
    app_ready = True
    if hasattr(startup_complete, 'set') and not startup_complete.is_set():
        startup_complete.set()
    return app

@pytest.mark.asyncio
async def test_new_session(mock_db):
    """Test creating a new session"""
    from app.main import app_ready
    from app.api.dependencies import startup_complete
    app_ready = True
    if hasattr(startup_complete, 'set') and not startup_complete.is_set():
        startup_complete.set()
    
    # Create a test client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First check the health endpoint
        health_response = await client.get("/health")
        print(f"Health check response: {health_response.json()}")
        
        # Try to create a new session
        response = await client.post("/new_session")
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        
        assert response.status_code == 200
        json_data = response.json()
        
        assert "sessionId" in json_data
        assert "chatId" in json_data
        
        session_id = json_data["sessionId"]
        
        # Get the session data
        response = await client.get(f"/session/{session_id}")
        assert response.status_code == 200
        session_data = response.json()
        
        assert "messages" in session_data
        assert len(session_data["messages"]) >= 1

@pytest.mark.asyncio
async def test_chat_endpoint(mock_db):
    """Test the chat endpoint"""
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
async def test_invalid_session(mock_db):
    """Test behavior with invalid session ID"""
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
