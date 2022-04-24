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

dag = DAG('python_dag_with_context',
          default_args=default_args,
          schedule_interval='30 0 * * *')


def print_current_date_provide_context(*args, **kwargs):
    """
    provide_context=True로 지정하면 kwargs 다양한 값들이 저장됨
    {'dag': <DAG: python_dag_with_jinja>,
    'ds': '2022-04-20',
    'next_ds': '2022-04-21',
    'next_ds_nodash': '20220421',
    'prev_ds': '2022-04-19',
    'prev_ds_nodash': '20220419',
    'ds_nodash': '20220420',
    'ts': '2022-04-20T00:30:00+00:00',
    'ts_nodash': '20220420T003000',
    'ts_nodash_with_tz': '20220420T003000+0000',
    'yesterday_ds': '2022-04-19',
    'yesterday_ds_nodash': '20240419',
    'tomorrow_ds': '2022-04-21',
    'tomorrow_ds_nodash': '20220421',
    'execution_date': <Pendulum [2022-04-20T00:30:00+00:00]> ...}
    """
    print(f"kwargs :{kwargs}")
    execution_date = kwargs['ds']
    execution_date = datetime.strptime(execution_date, "%Y-%m-%d").date()
    date_kor = ["월", "화", "수", "목", "금", "토", "일"]
    datetime_weeknum = execution_date.weekday()
    print(f"{execution_date}는 {date_kor[datetime_weeknum]}요일입니다")



python_task_context = PythonOperator(
    task_id='print_current_date_with_context_variable',
    python_callable=print_current_date_provide_context,
    provide_context=True,
    dag=dag,
)


python_task_context
