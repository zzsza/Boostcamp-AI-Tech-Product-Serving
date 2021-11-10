#### 1.2 dynamic logging config
import logging

dynamic_logger = logging.getLogger()
log_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(name) | %(levelname)s - %(message)s")
log_handler.setFormatter(formatter)
dynamic_logger.addHandler(log_handler)

dynamic_logger.info("hello world")
