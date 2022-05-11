#### 1.2 dynamic logging config
import logging
import sys

dynamic_logger = logging.getLogger()
dynamic_logger.setLevel(logging.DEBUG)
log_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_handler.setFormatter(formatter)
dynamic_logger.addHandler(log_handler)

dynamic_logger.info("hello world")
