from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
                   'owner': 'kyle',
                   'depends_on_past': False, # 이전 DAG의 Task가 성공, 실패했는지에 따라 현재 DAG 실행 여부가 결정. False는 과거 상관없이 매일 실행
                   'start_date': datetime(2022, 4, 20),
                   'retries': 1, # 실패시 재시도 횟수
                   'retry_delay': timedelta(minutes=5),
                   # 'priority_weight': 10, # DAG의 우선 순위를 설정할 수 있음
                   # 'end_date': datetime(2022, 4, 24), # 이 DAG을 마지막으로 실행할 date
                   # 'execution_timeout': timedelta(seconds=300), # 실행 타임아웃 : 300초 뒤에 오류 발생
                   # 'on_failure_callback': some_function, # 실패했을 때 실행할 함수
                   # 'on_success_callback': some_other_function, # 성공했을 때 실행할 함수
                   # 'on_retry_callback': another_function, # 재실행할 때 실행할 함수
                   # 'trigger_rule': 'all_success' # Task는 업스트림(앞선 Task)가 성공될 때만 실행됨. 이 조건을 수정할 수 있음
               },

# with 구문으로 DAG 정의
with DAG(
        dag_id='bash_dag',
        default_args=default_args,
        schedule_interval='@once',
        tags=['my_dags'],
) as dag:
    # BashOperator 사용
    task1 = BashOperator(
        task_id='print_date', # task의 id
        bash_command='date', # 실행할 bash command
        )

    task2 = BashOperator(
        task_id='sleep',
        bash_command='sleep 5',
        retries=2, # 만약 실패하면 재시도 2번 진행
        )

    task3 = BashOperator(
        task_id='pwd',
        bash_command='pwd',
        dag=dag)

    task1 >> task2 # task1 후에 task2 실행
    task1 >> task3 # task1 후에 task3 실행(2와 3은 병렬로 실행)

