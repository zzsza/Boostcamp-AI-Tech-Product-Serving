from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "kyle",
    "depends_on_past": False, # 이전 DAG의 Task 성공 여부에 따라서 현재 Task를 실행할지 말지가 결정. False는 과거 Task의 성공 여부와 상관없이 실행
    "start_date": datetime(2024, 1, 1),
    "end_date": datetime(2024, 1, 4),
    'retries': 1,  # 실패시 재시도 횟수
    'retry_delay': timedelta(minutes=5)  # 만약 실패하면 5분 뒤 재실행
}

def print_current_date():
    date_kor = ["월", "화", "수", "목", "금", "토", "일"]
    date_now = datetime.now().date()
    datetime_weeknum = date_now.weekday()
    print(f"{date_now}는 {date_kor[datetime_weeknum]}요일입니다")


with DAG(
    dag_id="python_dag1",
    default_args=default_args,
    schedule_interval="30 0 * * *", # UTC 시간 기준으로 매일 0시 30분에 실행하겠다. 한국 시간으로 9시 30분에 실행!
    tags=['my_dags'],
    catchup=True
) as dag:
    
    python_task = PythonOperator(
        task_id="print_current_date",
        python_callable=print_current_date
    )

    python_task
    