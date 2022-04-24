from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'kyle',
    'depends_on_past': False,
    'start_date': datetime(2022, 4, 20),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG('python_dag_with_jinja',
          default_args=default_args,
          schedule_interval='30 0 * * *')


def print_current_date_jinja(*args, **kwargs):
    """
    jinja template(today)가 templates_dict으로 저장되서 kwargs에서 사용할 수 있음
    """
    execution_date = kwargs.get('templates_dict').get('today', None)
    execution_date = datetime.strptime(execution_date, "%Y-%m-%d").date()
    date_kor = ["월", "화", "수", "목", "금", "토", "일"]
    datetime_weeknum = execution_date.weekday()
    print(f"{execution_date}는 {date_kor[datetime_weeknum]}요일입니다")


today = "{{ ds }}"

python_task_jinja = PythonOperator(
    task_id='print_current_date_with_jinja',
    python_callable=print_current_date_jinja,
    provide_context=True,
    templates_dict={
        'today': today,
    },
    dag=dag,
)


python_task_jinja