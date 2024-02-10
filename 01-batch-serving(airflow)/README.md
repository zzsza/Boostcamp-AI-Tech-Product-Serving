## Airflow
- Airflow를 처음 학습할 때는 Local 환경에서 DAG 작성에 집중하시는 것을 추천합니다
    - 그 후, 익숙해지면 Docker 환경에서 설정하고 DAG 작성을 해보시는 것을 추천합니다
    - Docker 환경을 설정하는 방법은 [Docker README](https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/blob/main/01-batch-serving(airflow)/docker-readme.md)에 작성해두었습니다

## 설치
- 가상 환경 설정

    ```
    python -m venv .venv
    source .venv/bin/activate
    ```

- Apache Airflow 설치

    ```
    pip3 install pip --upgrade

    AIRFLOW_VERSION=2.6.3
    PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
    CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

    pip3 install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
    ```

- Airflow DB init

    ```
    export AIRFLOW_HOME=`pwd`
    echo $AIRFLOW_HOME

    airflow db init
    ```

- Airflow Admin 생성

    ```
    airflow users create \
    --username admin \
    --password '!boostcamp-aitech!' \
    --firstname kyle \
    --lastname byeon \
    --role Admin \
    --email snugyun01@gmail.com 
    ```

- Airflow Webserver 실행

    ```
    airflow webserver --port 8080
    ```

- Airflow Scheduler 실행

    ```
    airflow scheduler
    ```


---

### 01-bash-operator
- 배우는 내용
    - Airflow에서 Bash 명령어 실행
    - Airflow Task 병렬(Parallel) 실행
    - Airflow 기초 내용 


### 02-python-operator
- 배우는 내용
    - Airflow에서 Python 함수 실행

### 03-python-operator-with-context
- 배우는 내용
    - Airflow에서 Python 함수 실행
    - Airflow에서 DAG이 실행되는 시간과 연관해서 실행하는 방법에 대해 학습
    - 예 : 20220402 날짜로 실행하면 해당 날짜의 데이터를 처리하고, 20220403 날짜로 실행하면 해당 날짜의 데이터를 처리

### 04-python-operator-with-jinja
- 배우는 내용
    - 03에서 학습한 내용을 Jinja Template을 활용해 구현
- 참고 자료
    - [Airflow Jinja Template](https://airflow.apache.org/docs/apache-airflow/stable/templates-ref.html)

### 05-python-operator-with-slack-noti
- 배우는 내용
  - DAG의 Task가 실패할 때 슬랙 메세지를 전송합니다

### 06-simpe_elt.py
- 배우는 내용
  - 간단한 ELT 파이프라인을 만들고, 데이터 엔지니어링에 대해 연습합니다
  - 이 코드는 Google Cloud Composer에서 실행합니다


### 추가 학습 자료
- Local에서 위 실습을 모두 완료했다면, Docker 환경에서 실행해보는 것을 추천합니다(강의에서는 따로 다루진 않지만 꼭 실행해보세요) - [Docker README](https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/blob/main/01-batch-serving(airflow)/docker-readme.md)
- 그 이후엔 멘토 전시흠님이 만드신 [Apache Airflow Tutorials for Beginner](https://heumsi.github.io/apache-airflow-tutorials-for-beginner/)을 처음부터 보면서 따라가시면 Airflow를 더 잘 이해할 수 있을거에요 :) 
- 꼭 해보길 추천하는 부분(검색해서 추가 학습해서 개인 블로그에 정리하면 Best)
    - Airflow Variables
    - Airflow Task Fail인 경우 슬랙 메세지 보내기
    - 조건에 따라 다른 함수 실행하기 : BranchPythonOperator
    - Task 끼리 데이터를 주고 받아야 하는 경우 : XCom
- [Apache Airflow 기반의 데이터 파이프라인](http://www.yes24.com/Product/Goods/107878326) 이 책도 매우 추천합니다! :) 