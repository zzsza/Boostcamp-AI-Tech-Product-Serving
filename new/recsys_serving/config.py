from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    db_url: str = Field(default="sqlite:///./db.sqlite3", env="DB_URL")
    data_path: str = Field(default="data_path", env="DATA_PATH")
    model_path: str = Field(default="model_path", env="MODEL_PATH")


config = Config()
