from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import AsyncMock, patch

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_login_route_success():
    with patch("api.main.HNClient") as MockClass:
        m_client = MockClass.return_value
        m_client.login = AsyncMock(return_value=(True, "OK"))
        m_client.close = AsyncMock()
        
        response = client.post("/login", json={"username": "u", "password": "p"})
        assert response.status_code == 200
        assert response.json()["username"] == "u"

def test_login_route_failure():
    with patch("api.main.HNClient") as MockClass:
        m_client = MockClass.return_value
        m_client.login = AsyncMock(return_value=(False, "Error"))
        m_client.close = AsyncMock()
        
        response = client.post("/login", json={"username": "u", "password": "p"})
        assert response.status_code == 401

def test_vote_route_success():
    with patch("api.main.HNClient") as MockClient:
        m_client = MockClient.return_value
        m_client.vote = AsyncMock(return_value=(True, "OK"))
        m_client.close = AsyncMock()
        m_client.__aenter__ = AsyncMock(return_value=m_client)
        m_client.__aexit__ = AsyncMock()
        
        response = client.post("/vote", json={"story_id": 123, "direction": "up"})
        assert response.status_code == 200
        assert response.json() == {"status": "success"}
