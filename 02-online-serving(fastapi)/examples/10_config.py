import os
from typing import Dict, Any

dev_config_yaml = """
env: dev
db:
  username: user
  password: user
  host: localhost
  port: 3306
  database: dev
"""
with open("dev_config.yaml", "w") as f:
    f.write(dev_config_yaml)

# 1. .ini, .yaml 파일 기반 config 주입
from yaml import load, FullLoader


def load_config(config_path: str) -> Dict[str, Any]:
    """
    config YAML 파일을 로드합니다

    Args:
        config_path: config YAML 파일 경로

    Returns:
        Dict[str, Any]: Config dictionary
    """
    with open(config_path, "r") as f:
        config = load(f, FullLoader)
    return config


config = load_config(config_path="dev_config.yaml")

assert config["env"] == "dev"
expected = {"username": "user", "password": "user", "host": "localhost", "port": 3306, "database": "dev"}
assert config["db"] == expected


# 2. flask style config
class Config(object):
    ENV: str = None
    TESTING: bool = False
    DB: Dict[str, Any] = {}

    @classmethod
    def from_yaml(cls, config_path: str):
        import yaml

        with open(config_path, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)

        cls.ENV = config["env"]
        cls.DB = config["db"]
        return cls


class DevConfig(Config):
    pass


class ProdConfig(Config):
    pass


config = DevConfig.from_yaml("dev_config.yaml")
assert config.ENV == "dev"
assert config.DB == expected

# 3. pydantic base settings
from pydantic import BaseSettings, Field
from enum import Enum


class ConfigEnv(str, Enum):
    DEV = "dev"
    PROD = "prod"


class DBConfig(BaseSettings):
    host: str = Field(default="localhost", env="db_host")
    port: int = Field(default=3306, env="db_port")
    username: str = Field(default="user", env="db_username")
    password: str = Field(default="user", env="db_password")
    database: str = Field(default="dev", env="db_database")


class AppConfig(BaseSettings):
    env: ConfigEnv = Field(default="dev", env="env")
    db: DBConfig = DBConfig()


with open("dev_config.yaml", "r") as f:
    config = load(f, FullLoader)

config_with_pydantic = AppConfig(**config)

assert config_with_pydantic.env == "dev"
assert config_with_pydantic.db.dict() == expected

# 환경 변수로 필드를 오버라이딩합니다.
os.environ["ENV"] = "prod"
os.environ["DB_HOST"] = "mysql"
os.environ["DB_USERNAME"] = "admin"
os.environ["DB_PASSWORD"] = "SOME_SAFE_PASSWORD"

prod_config_with_pydantic = AppConfig()
assert prod_config_with_pydantic.env == "prod"
assert prod_config_with_pydantic.dict() != expected

# cleanup
os.remove("dev_config.yaml")
