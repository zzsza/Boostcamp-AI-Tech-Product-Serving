from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'kyle',
    'depends_on_past': False,
    'start_date': datetime(2022, 4, 20),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG('python_dag1',
          default_args=default_args,
          schedule_interval='30 0 * * *')


def print_current_date():
    date_kor = ["월", "화", "수", "목", "금", "토", "일"]
    date_now = datetime.now().date()
    datetime_weeknum = date_now.weekday()
    print(f"{date_now}는 {date_kor[datetime_weeknum]}요일입니다")


python_task = PythonOperator(
    task_id='print_current_date',
    python_callable=print_current_date,
    dag=dag,
)

python_task
