from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "kyle",
    "depends_on_past": False, # 이전 DAG의 Task 성공 여부에 따라서 현재 Task를 실행할지 말지가 결정. False는 과거 Task의 성공 여부와 상관없이 실행
    "start_date": datetime(2024, 1, 1)
}

with DAG(
    dag_id="bash_dag", 
    default_args=default_args,
    schedule_interval="@once",
    tags=["my_dags"]
) as dag:
    
    task1 = BashOperator(
        task_id="print_date", # task의 id
        bash_command="date" # 실행할 bash command를 저장
    )

    task2 = BashOperator(
        task_id="sleep",
        bash_command="sleep 5",
        retries=2 # 만약 bash command가 실패하면 2번 재시도
    )

    task3 = BashOperator(
        task_id='pwd',
        bash_command='pwd'
    )

    task1 >> task2 # task1이 완료되면, task2를 실행
    task1 >> task3 # task1이 완료되면, task3을 실행
    # task1 >> [task2, task3]