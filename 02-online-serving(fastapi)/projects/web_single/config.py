from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    db_url: str = Field(default="sqlite:///./db.sqlite3", env="DB_URL")
    model_path: str = Field(default="model.joblib", env="MODEL_PATH")


config = Config()
