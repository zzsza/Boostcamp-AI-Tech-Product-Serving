from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_api():
    response = client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}
