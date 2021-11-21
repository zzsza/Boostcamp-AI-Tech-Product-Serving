import json
import logging.config
import os
from typing import Union

from pydantic import BaseModel, BaseSettings, Field
from datetime import datetime
from logging import StreamHandler, LogRecord

import pytz
import yaml
from google.cloud import bigquery
from google.oauth2 import service_account
from pythonjsonlogger import jsonlogger

log_format = ", ".join(
    [
        f"%({key})s"
        for key in sorted(["filename", "levelname", "name", "message", "created"])
    ]
)


class BigqueryLogSchema(BaseModel):
    level: str
    message: str
    created_at: datetime


class BigqueryHandlerConfig(BaseSettings):
    credentials: service_account.Credentials
    table: Union[str, bigquery.TableReference]
    formatter: logging.Formatter = Field(default_factory=jsonlogger.JsonFormatter)
    level: int = Field(default=logging.INFO)

    class Config:
        arbitrary_types_allowed = True


class BigqueryHandler(StreamHandler):
    def __init__(self, config: BigqueryHandlerConfig) -> None:
        super().__init__()
        self.config = config
        self.bigquery_client = bigquery.Client(credentials=self.config.credentials)
        self.setLevel(config.level)
        self.setFormatter(fmt=self.config.formatter)

    def emit(self, record: LogRecord) -> None:
        message = self.format(record)
        json_message = json.loads(message)
        log_input = BigqueryLogSchema(
            level=json_message["levelname"],
            message=json_message["message"],
            created_at=datetime.fromtimestamp(
                json_message["created"], tz=pytz.timezone("Asia/Seoul")
            ),
        )
        errors = self.bigquery_client.insert_rows_json(
            self.config.table, [json.loads(log_input.json())]
        )
        if errors:
            print(errors)  # 에러가 발생해도 Logging이 정상적으로 동작하게 하기 위해, 별도의 에러 핸들링을 추가하지 않습니다


def get_ml_logger(
    config_path: Union[os.PathLike, str],
    credential_json_path: Union[os.PathLike, str],
    table_ref: Union[bigquery.TableReference, str],
    logger_name: str = "MLLogger",
) -> logging.Logger:
    """
    MLLogger를 가져옵니다

    Args:
        config_path: logger config YAML 파일의 경로
        credential_json_path: service account json 파일 경로
        table_ref: 로그가 저장될 빅쿼리의 table reference (e.g., project.dataset.table_name)
        logger_name: [optional] logger의 이름(default: MLLogger)

    Returns:
        logging.Logger: MLLogger

    """
    # Default Logger Config를 추가합니다
    with open(config_path, "r") as f:
        logging_config = yaml.safe_load(f)
    logging.config.dictConfig(logging_config)
    _logger = logging.getLogger(logger_name)

    # BigQuery Logging Handler 추가합니다
    if not credential_json_path:
        return _logger

    credentials = service_account.Credentials.from_service_account_file(
        filename=credential_json_path
    )
    bigquery_handler_config = BigqueryHandlerConfig(
        credentials=credentials,
        table=table_ref,
        formatter=jsonlogger.JsonFormatter(fmt=log_format),
    )
    bigquery_handler = BigqueryHandler(config=bigquery_handler_config)
    _logger.addHandler(bigquery_handler)

    return _logger


if __name__ == "__main__":
    from pathlib import Path

    here = Path(__file__)
    config_yaml_path = os.path.join(here.parent, "config.yaml")

    logger = get_ml_logger(
        config_path=config_yaml_path,
        credential_json_path="서비스 계정 JSON 파일 경로",  # FIXME
        table_ref="빅쿼리 테이블 주소",  # FIXME: e.g., boostcamp-ai-tech-serving.online_serving_logs.mask_classification
    )
    for _ in range(10):
        logger.info("hello world")
