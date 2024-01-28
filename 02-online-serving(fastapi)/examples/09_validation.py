class ModelInput01:
    def __init__(self, url: str, rate: int, target_dir: str) -> None:
        self.url = url
        self.rate = rate
        self.target_dir = target_dir

    def _validate_url(self, url_like: str) -> bool:
        """
        올바른 url인지 검증합니다

        Args:
            url_like (str): 검증할 url

        Returns:
            bool: 검증 성공/실패 여부
        """
        from urllib.parse import urlparse

        try:
            result = urlparse(url_like)
            return all([result.scheme, result.netloc])
        except:
            return False

    def _validate_directory(self, directory_like: str) -> bool:
        """
        존재하는 디렉토리인지 검증합니다

        Args:
            directory_like (str): 검증할 디렉토리 경로

        Returns:
            bool: 검증 성공/실패 여부
        """
        import os

        return os.path.isdir(directory_like)

    def validate(self) -> bool:
        """
        클래스 필드가 올바른지 검증합니다.

        Returns:
            bool: 검증/성공 실패 여부
        """
        validation_results = [
            self._validate_url(self.url),
            1 <= self.rate <= 10,
            self._validate_directory(self.target_dir),
        ]
        return all(validation_results)


from dataclasses import dataclass

from pydantic.networks import HttpUrl


class ValidationError(Exception):
    pass


@dataclass
class ModelInput02:
    url: str
    rate: int
    target_dir: str

    def _validate_url(self, url_like: str) -> bool:
        """
        올바른 url인지 검증합니다

        Args:
            url_like (str): 검증할 url

        Returns:
            bool: 검증 성공/실패 여부
        """
        from urllib.parse import urlparse

        try:
            result = urlparse(url_like)
            return all([result.scheme, result.netloc])
        except:
            return False

    def _validate_directory(self, directory_like: str) -> bool:
        """
        존재하는 디렉토리인지 검증합니다

        Args:
            directory_like (str): 검증할 디렉토리 경로

        Returns:
            bool: 검증 성공/실패 여부
        """
        import os

        return os.path.isdir(directory_like)

    def validate(self) -> bool:
        """
        클래스 필드가 올바른지 검증합니다.

        Returns:
            bool: 검증/성공 실패 여부
        """
        validation_results = [
            self._validate_url(self.url),
            1 <= self.rate <= 10,
            self._validate_directory(self.target_dir),
        ]
        return all(validation_results)

    def __post_init__(self):
        if not self.validate():
            raise ValidationError("올바르지 않은 input 입니다")


from pydantic import BaseModel, HttpUrl, Field, DirectoryPath


class ModelInput03(BaseModel):
    url: HttpUrl
    rate: int = Field(ge=1, le=10)
    target_dir: DirectoryPath


if __name__ == "__main__":
    import os

    VALID_INPUT = {
        "url": "https://content.presspage.com/uploads/2658/c800_logo-stackoverflow-square.jpg?98978",
        "rate": 4,
        "target_dir": os.path.join(os.getcwd(), "examples"),
    }

    INVALID_INPUT = {"url": "WRONG_URL", "rate": 11, "target_dir": "WRONG_DIR"}

    valid_python_class_model_input = ModelInput01(**VALID_INPUT)
    assert valid_python_class_model_input.validate() is True

    invalid_python_class_model_input = ModelInput01(**INVALID_INPUT)
    assert invalid_python_class_model_input.validate() is False

    valid_dataclass_model_input = ModelInput02(**VALID_INPUT)
    assert valid_dataclass_model_input.validate() is True

    try:
        invalid_dataclass_model_input = ModelInput02(**INVALID_INPUT)  # Error
    except ValidationError as exc:
        print("dataclass model input validation error", str(exc))
        pass

    from pydantic import ValidationError

    valid_pydantic_model_input = ModelInput03(**VALID_INPUT)
    try:
        invalid_pydantic_model_input = ModelInput03(**INVALID_INPUT)  # error
    except ValidationError as exc:
        print("pydantic model input validation error: ", exc.json())
        pass

    # Expected:
    # dataclass model input validation error 올바르지 않은 input 입니다
    # pydantic model input validation error:  [
    #   {
    #     "loc": [
    #       "url"
    #     ],
    #     "msg": "invalid or missing URL scheme",
    #     "type": "value_error.url.scheme"
    #   },
    #   {
    #     "loc": [
    #       "rate"
    #     ],
    #     "msg": "ensure this value is less than or equal to 10",
    #     "type": "value_error.number.not_le",
    #     "ctx": {
    #       "limit_value": 10
    #     }
    #   },
    #   {
    #     "loc": [
    #       "target_dir"
    #     ],
    #     "msg": "file or directory at path \"WRONG_DIR\" does not exist",
    #     "type": "value_error.path.not_exists",
    #     "ctx": {
    #       "path": "WRONG_DIR"
    #     }
    #   }
    # ]
