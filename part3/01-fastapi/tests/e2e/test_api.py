import os
import uuid
from pathlib import Path

import pytest
import typing as T
from fastapi.testclient import TestClient
from fastapi import status

import app.main

HERE = Path(__file__)
TEST_INPUT_DIR = os.path.join(HERE.parent, "images")


def get_test_image_input(
        filename: str, base_dir: str = TEST_INPUT_DIR
) -> T.Tuple[str, T.BinaryIO, str]:
    if not os.path.isdir(base_dir):
        raise NotADirectoryError()

    image_input_path = os.path.join(base_dir, filename)
    if not os.path.exists(image_input_path):
        raise FileNotFoundError()

    return filename, open(image_input_path, "rb"), "image/jpeg"


@pytest.fixture
def client():
    return TestClient(app=app.main.app)


def validate_uuid(val: T.Any) -> bool:
    try:
        uuid_obj = uuid.UUID(str(val), version=4)
    except ValueError:
        return False
    return str(uuid_obj) == val


@pytest.mark.parametrize(
    ["endpoint", "method", "req_body"],
    [
        (
                "/order",
                "POST",
                {"files": get_test_image_input(filename="mask_input_1.jpg")},
        ),
        (
                "/order",
                "POST",
                {"files": get_test_image_input(filename="mask_input_2.jpg")},
        ),
        (
                "/order",
                "POST",
                [
                    ("files", get_test_image_input(filename="mask_input_1.jpg")),
                    ("files2", get_test_image_input(filename="mask_input_2.jpg")),
                ],
        ),
    ],
)
def test_make_order_return_order_id(
        endpoint: str,
        method: str,
        req_body: T.Optional[T.Dict[str, T.Any]],
        client: TestClient,
):
    # given

    # when
    with client:
        response = client.request(method=method, url=endpoint, files=req_body)

    # then
    assert validate_uuid(response.json()) is True


def test_make_order_no_file_should_return_422_error(client: TestClient):
    # given
    no_file_payload = {}

    # when
    with client:
        response = client.post("/order", files=no_file_payload)

    # then
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_get_order_not_existing_id_should_return_dict(client: TestClient):
    # given
    random_uuid = uuid.uuid4()

    # when
    with client:
        response = client.get(f"/order/{random_uuid}")

    # then
    expected = {"message": "주문 정보를 찾을 수 없습니다"}
    assert response.json() == expected
