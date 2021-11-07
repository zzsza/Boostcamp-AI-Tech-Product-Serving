# Logging
ML Online Serving - 서빙 API 로그 데이터를 쌓습니다

## Implementation
python logging 모듈을 활용해서 console 출력, 파일, 그리고 bigquery table로 로그 데이터를 적재합니다 

## 실행 방법
### 환경 변수
| KEY | VALUE |  
| ---------------|----------------|
| LOGGING_CONFIG_YAML_PATH | config.yaml의 경로 | 
| LOGGING_BIGQUERY_CREDENTIAL_JSON_PATH | bigquery 서비스 계정 json 파일 경로 |
| LOGGING_BIGQUERY_TABLE_REF | 로그 데이터를 적재할 bigquery table reference (e.g., project.dataset.table) |
**위의 환경 변수를 반드시 지정해야합니다**

### 실행 방법 
```shell
from app.logger import get_ml_logger

logger = get_ml_logger(config_path=os.getenv("LOGGING_CONFIG_YAML_PATH"),
                       credential_json_path=os.getenv("LOGGING_BIGQUERY_CREDENTIAL_JSON_PATH"),
                       table_ref=os.getenv("LOGGING_BIGQUERY_TABLE_REF"))
logger.info("hello world") 
```
