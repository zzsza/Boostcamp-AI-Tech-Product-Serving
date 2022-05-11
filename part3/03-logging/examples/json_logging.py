#### JSON Logging
import logging

from pythonjsonlogger import jsonlogger


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

logger.info("hello world")
