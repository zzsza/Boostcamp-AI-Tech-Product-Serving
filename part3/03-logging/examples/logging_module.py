#### 1. logging module 써보기
import logging

logger = logging.getLogger("example")  # root logger
logger.info("hello world")  # 아무런 로그도 출력되지 않습니다.

#### 1.1 logging module config 추가하기
import logging.config

logger_config = {
    "version": 1,  # required
    "disable_existing_loggers": True,  # 다른 Logger를 overriding 합니다
    "formatters": {
        "simple": {"format": "%(asctime)s | %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "simple",
        }
    },
    "loggers": {"example": {"level": "INFO", "handlers": ["console"]}},
}

logging.config.dictConfig(logger_config)
logger_with_config = logging.getLogger("example")
logger_with_config.info("이제는 보이죠?")
