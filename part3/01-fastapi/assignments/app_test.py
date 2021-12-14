import pytest
from fastapi.testclient import TestClient
from .app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_get_models_should_return_empty_list_when_no_model_exists(client):
    response = client.get("/models")
    assert response.json() == []


def test_get_model_should_return_should_respond_404_when_no_model_exists(client):
    response = client.get("/model/999999")
    assert response.status_code == 404


def test_create_model_should_append_new_model_to_list(client):
    client.post("/model", json={
        "name": "test_model",
        "version": 1,
        "description": "test model 입니다",
        "tags": [],
        "artifact_url": "https://boostcamp.connect.or.kr/"
    })

    models_response = client.get("/models")
    assert len(models_response.json()) == 1


def test_create_model_should_respond_new_model_id(client):
    response = client.post("/model", json={
        "name": "test_model_2",
        "version": 1,
        "description": "test model 2 입니다",
        "tags": [],
        "artifact_url": "https://boostcamp.connect.or.kr/"
    })
    assert response.json() == 2  # TODO: 멱등하게 바꾸기


def test_update_model_should_replace_existing_model_in_list(client):
    client.patch("/model/2", json={
        "version": 2,
        "tags": ["updated"],
        "artifact_url": "https://boostcamp.connect.or.kr/about.html"
    })

    models_response = client.get("/model?model_name=test_model_2")
    assert models_response.json() == {
        "id": 2,
        "name": "test_model_2",
        "description": "test model 2 입니다",
        "version": '2',
        "tags": ["updated"],
        "artifact_url": "https://boostcamp.connect.or.kr/about.html"
    }


def test_update_model_should_respond_404_when_no_matching_model_exists(client):
    response = client.patch("/model/99999", json={
        "version": 2,
        "tags": ["updated"],
        "artifact_url": "https://boostcamp.connect.or.kr/about.html"
    })
    assert response.status_code == 404
    assert response.json()['detail'] == "모델을 찾을 수 없습니다"


def test_delete_model_should_respond_204_when_succeeds(client):
    response = client.delete("/model/2")

    assert response.status_code == 204
