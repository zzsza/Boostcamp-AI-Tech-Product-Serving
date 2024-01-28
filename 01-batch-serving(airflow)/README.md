## Airflow
- Airflow를 처음 학습할 때는 Local 환경에서 DAG 작성에 집중하시는 것을 추천합니다
    - 그 후, 익숙해지면 Docker 환경에서 설정하고 DAG 작성을 해보시는 것을 추천합니다
    - Docker 환경을 설정하는 방법은 [Docker README](https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/blob/main/part4/02-airflow/docker-readme.md)에 작성해두었습니다
    - Docker 환경에선 VS Code Server를 활용하는 것도 추가했으니, 이 부분도 꼭 보시는 것을 추천드립니다


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


### 추가 학습 자료
- Local에서 위 실습을 모두 완료했다면, Docker 환경에서 실행해보는 것을 추천합니다(강의에서는 따로 다루진 않지만 꼭 실행해보세요) - [Docker README](https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/blob/main/part4/02-airflow/docker-readme.md)
- 그 이후엔 멘토 전시흠님이 만드신 [Apache Airflow Tutorials for Beginner](https://heumsi.github.io/apache-airflow-tutorials-for-beginner/)을 처음부터 보면서 따라가시면 Airflow를 더 잘 이해할 수 있을거에요 :) 
- 꼭 해보길 추천하는 부분(검색해서 추가 학습해서 개인 블로그에 정리하면 Best)
    - Airflow Variables
    - Airflow Task Fail인 경우 슬랙 메세지 보내기
    - 조건에 따라 다른 함수 실행하기 : BranchPythonOperator
    - Task 끼리 데이터를 주고 받아야 하는 경우 : XCom
- [Apache Airflow 기반의 데이터 파이프라인](http://www.yes24.com/Product/Goods/107878326) 이 책도 매우 추천합니다! :) 