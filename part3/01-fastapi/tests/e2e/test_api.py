from fastapi.testclient import TestClient
import pytest

from assignments.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_get_models_should_return_empty_list_when_no_model_exists(client):
    assert False


def test_get_model_should_return_should_respond_404_when_no_model_exists(client):
    assert False


def test_create_model_should_append_new_model_to_list(client):
    assert False


def test_create_model_should_respond_new_model_id(client):
    assert False


def test_update_model_should_replace_existing_model_in_list(client):
    assert False


def test_update_model_should_respond_404_when_no_matching_model_exists(client):
    assert False


def test_delete_model_should_respond_204_when_succeeds(client):
    assert False
