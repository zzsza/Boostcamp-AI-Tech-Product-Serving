# Airflow
- Airflow를 처음 학습할 때는 Local 환경에서 DAG 작성에 집중하시는 것을 추천합니다
    - 그 후, 익숙해지면 Docker 환경에서 설정하고 DAG 작성을 해보시는 것을 추천합니다
    - Docker 환경을 설정하는 방법은 [Docker README](https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/blob/main/01-batch-serving(airflow)/docker-readme.md)에 작성해두었습니다
- 실습을 할 때, git pull로 진행하지 말고 여러분들의 폴더를 만들어서 하나씩 작업하는 것을 추천합니다.
    - 서버 환경보다 Local 환경을 추천합니다. 


## 설치
- 사용하는 Ariflow의 버전은 2.6.3으로, Python 3.7 ~ 3.11만 지원하니 Python 버전을 확인해주세요. 

### 가상 환경 설정

```
python -m venv .venv
source .venv/bin/activate
```

### Apache Airflow 설치
- 그냥 설치할 경우 의존성 이슈로 오류가 발생할 수 있기 때문에 아래 명령어로 Airflow를 설치합니다.

```
pip3 install pip --upgrade

AIRFLOW_VERSION=2.6.3
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)" 
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

pip3 install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
```



## Airflow DB init

```
export AIRFLOW_HOME=$(pwd)
echo $AIRFLOW_HOME
export TZ=UTC

airflow db init
```

## Airflow Admin 생성

```
airflow users create \
--username admin \
--password '!boostcamp-aitech!' \
--firstname kyle \
--lastname byeon \
--role Admin \
--email snugyun01@gmail.com 
```

## Airflow Webserver 실행
- Airflow는 Webserver, Scheduler를 모두 실행해야 정상적으로 작동합니다

```
airflow webserver --port 8080
```

- 만약 아래와 같은 오류가 발생한다면, `airflow.cfg` 파일을 수정하면 됩니다(오류가 발생하지 않으면 다음 단계로)
    - `airflow.cfg` 파일의 `sql_alchemy_conn` 값을 절대 경로로 수정하면 됩니다. `sqlite:///./airflow.db`로 수정하면 됩니다
        - 예 : `sql_alchemy_conn = sqlite:///${AIRFLOW_HOME}/airflow.db`

    ```
    airflow.exceptions.AirflowConfigException: Cannot use relative path: `sqlite:///./airflow.db` to connect to sqlite. Please use absolute path such as `sqlite:////tmp/airflow.db
    ```

### Airflow Scheduler 실행
```
airflow scheduler
```

- Database init 내용이 나올 때 y를 입력

```
Please confirm database initialize (or wait 4 seconds to skip it). Are you sure? [y/N] 
```

---

# Airflow 실습
## 01-bash-operator
- 실습 개요 : Airflow의 가장 기본적인 Operator인 BashOperator를 사용하여 간단한 쉘 명령어를 실행하는 DAG를 작성합니다
- 실습 목적 : Airflow DAG의 기본 구조(Task 정의, 의존성 설정)를 이해하고, Task가 병렬로 실행되는 원리를 체득합니다
- 배우는 내용
    - DAG 작성 방법 및 구조 이해 
    - Task 간 의존성 설정(`>>`) 및 병렬 실행 개념 습득

## 02-python-operator
- 실습 개요: PythonOperator를 사용하여 파이썬 함수를 Airflow Task로 실행해봅니다
- 실습 목적: 데이터 처리에서 가장 많이 쓰이는 Python 코드를 Airflow 내에서 어떻게 실행하는지 익힙니다
- 배우는 내용
    - Python 함수와 Airflow Task의 연결 방법
    - Airflow 환경 내에서의 파이썬 코드 실행

## 03-python-operator-with-context
- 실습 개요: Airflow가 제공하는 Context 변수(실행 날짜, Task ID 등)를 파이썬 함수 내에서 활용하는 방법을 배웁니다
- 실습 목적: "오늘 날짜"가 아닌, "논리적 실행 기준 날짜(Data Interval)"를 기준으로 데이터를 처리하는 멱등성(Idempotency)의 개념을 이해합니다
- 배우는 내용
    - Airflow Context 및 Jinja Template 변수 활용 능력
    - 논리적 실행 날짜(`execution_date`, `data_interval_start`)에 대한 이해

## 04-python-operator-with-jinja
- 실습 개요: 파이썬 코드 내부가 아닌, Operator의 파라미터 레벨에서 Jinja Template을 사용하여 동적인 값을 주입합니다
- 실습 목적: 코드의 수정 없이 템플릿만으로 유연하게 변하는 파라미터를 처리하는 고급 기법을 익힙니다
- 배우는 내용
    - Jinja Template 문법 실전 활용
    - 동적 파라미터 주입을 통한 DAG 재사용성 증대
- 참고 자료
    - [Airflow Jinja Template](https://airflow.apache.org/docs/apache-airflow/stable/templates-ref.html)

## 05-python-operator-with-slack-noti
- 실습 개요: DAG 실행 중 실패가 발생했을 때, Slack으로 알람을 보내는 기능을 구현합니다
- 실습 목적: 운영 환경에서 필수적인 모니터링 및 알람 시스템을 구축하는 방법을 배웁니다
- 배우는 내용
    - 외부 Connection(Slack) 연동 및 관리
    - Callback 함수(`on_failure_callback`)를 이용한 에러 핸들링

### 필요 라이브러리 설치
```
pip3 install apache-airflow-providers-slack==8.6.0 
```


## 06-simple_elt.py
- 실습 개요: 실제 데이터를 추출(Extract), 적재(Load), 변환(Transform)하는 간단한 파이프라인을 구축합니다. (Google Cloud 연동)
- 실습 목적: 단순 작업 단위가 아닌, 데이터의 흐름(Pipeline)을 구현합니다. 이를 통해 간단한 데이터 엔지니어링 작업을 경험합니다.
- 배우는 내용
    - 기본적인 ELT 데이터 엔지니어링 프로세스 이해
    - Cloud Provider(GCP)와 Airflow의 통합 활용 능력
    - Google Cloud Composer 환경에서의 실행 경험

### 필요 라이브러리 설치
```
pip3 install apache-airflow-providers-google==10.14.0 
```

# 추가 학습 자료
- Local에서 위 실습을 모두 완료했다면, Docker 환경에서 실행해보는 것을 추천합니다(강의에서는 따로 다루진 않지만 꼭 실행해보세요)
    - [Docker README](https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/blob/main/01-batch-serving(airflow)/docker-readme.md)



## 추가적으로 학습하면 좋은 부분(추가 학습해서 개인 블로그에 정리하면 Best)
- Airflow Variables
- 조건에 따라 다른 함수 실행하기 : BranchPythonOperator
- Task 끼리 데이터를 주고 받아야 하는 경우 : XCom
- 현재 Airflow 버전은 2.6.3인데 25년 12월 기준 Airflow는 3점대 버전이 나왔습니다. 이런 경우 회사에서 2점대에서 3점대로 마이그레이션을 고민합니다. 
    - 2점대로 DAG을 만든 후, 3점대로 업그레이드를 하면서 DAG 파일을 마이그레이션하는 경험을 하는 것을 추천합니다. 이런 버전 마이그레이션 경험은 실무에서는 자주 겪는 일인데, 2점대를 3점대로 수정하는 경험을 하고 블로그에 잘 정리해보는 것을 추천합니다