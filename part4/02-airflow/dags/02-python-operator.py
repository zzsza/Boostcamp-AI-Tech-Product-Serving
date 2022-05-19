from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'kyle',
    'depends_on_past': False,  # 이전 DAG의 Task가 성공, 실패 여부에 따라 현재 DAG 실행 여부가 결정. False는 과거의 실행 결과 상관없이 매일 실행한다
    'start_date': datetime(2022, 4, 20),
    'retires': 1,  # 실패시 재시도 횟수
    'retry_delay': timedelta(minutes=5)  # 만약 실패하면 5분 뒤 재실행
}


# 사용할 함수 정의
def print_current_date():
    date_kor = ["월", "화", "수", "목", "금", "토", "일"]
    date_now = datetime.now().date()
    datetime_weeknum = date_now.weekday()
    print(f"{date_now}는 {date_kor[datetime_weeknum]}요일입니다")


# with 구문으로 DAG 정의
with DAG(
        dag_id='python_dag1',
        default_args=default_args,
        schedule_interval='30 0 * * *',  # UTC 시간 기준 0시 30분에 Daily로 실행하겠다! 한국 시간 기준 오전 9시 30분
        tags=['my_dags']
) as dag:
    python_task = PythonOperator(
        task_id='print_current_date',
        python_callable=print_current_date  # 실행할 python 함수
    )

    python_task
