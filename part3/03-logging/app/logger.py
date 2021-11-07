import json
import logging.config
import os
from dataclasses import dataclass, field
from datetime import datetime
from logging import StreamHandler, LogRecord

import yaml
from google.cloud import bigquery
from google.oauth2 import service_account
from pythonjsonlogger import jsonlogger


@dataclass
class BigqueryLogSchema:
    filename: str = field(default=None)
    levelname: str = field(default=None)
    name: str = field(default=None)
    message: str = field(default=None)
    asctime: datetime = field(default_factory=datetime.now)

    def to_log_format(self):
        return ", ".join([f"%({key})s" for key in sorted(self.__dict__.keys())])


@dataclass
class BigqueryHandlerConfig:
    credentials: service_account.Credentials
    table: bigquery.TableReference
    formatter: logging.Formatter = jsonlogger.JsonFormatter()
    level: int = logging.INFO


class BigqueryHandler(StreamHandler):
    def __init__(self, config: BigqueryHandlerConfig):
        StreamHandler.__init__(self)
        self.config = config
        self.bigquery_client = bigquery.Client(credentials=self.config.credentials)
        self.setLevel(config.level)
        self.setFormatter(fmt=self.config.formatter)

    def emit(self, record: LogRecord) -> None:
        message = self.format(record)
        json_message = json.loads(message)
        errors = self.bigquery_client.insert_rows(self.config.table, [json_message])
        if errors:
            print(errors)


def get_ml_logger(
    config_path: os.PathLike,
    credential_json_path: os.PathLike,
    table_ref: bigquery.TableReference,
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
        formatter=jsonlogger.JsonFormatter(fmt=BigqueryLogSchema().to_log_format()),
    )
    bigquery_handler = BigqueryHandler(config=bigquery_handler_config)
    _logger.addHandler(bigquery_handler)

    return _logger


if __name__ == "__main__":
    logger = get_ml_logger(
        config_path=os.getenv("LOGGING_CONFIG_YAML_PATH"),
        credential_json_path=os.getenv("LOGGING_BIGQUERY_CREDENTIAL_JSON_PATH"),
        table_ref=os.getenv("LOGGING_BIGQUERY_TABLE_REF"),
    )
    for _ in range(10):
        logger.info("hello world")
