from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# 앞선 02-python-operator.py는 "date_now = datetime.now().date()"를 사용하기 때문에
# 언제 실행해도 실행하는 시간 기준으로 실행됨
# Airflow는 Batch 성으로 특정 시간대로 실행하는 도구인데, 위와 같이 now 등을 사용하면 실행하는 시점이 아닌 현재 시점으로 실행됨
# Airflow에선 항상 현재 최신 작업만 실행하는 것이 아니고, 과거 날짜로 실행해야 하는 경우도 존재(Backfill이란 용어 사용)
# 따라서 코드 상에서 now(), current_datetime() 등을 사용하지 않고, Airflow에서 실행하기로 했던 시간을 넣어줘야 함
# 이를 execution_date라고 부름


default_args = {
                   'owner': 'kyle',
                   'depends_on_past': False,
                   'start_date': datetime(2022, 4, 20),
                   'retries': 1,
                   'retry_delay': timedelta(minutes=5),
               },


# 사용할 함수 정의
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


# with 구문으로 DAG 정의
with DAG(
        dag_id='python_dag_with_context',
        default_args=default_args,
        schedule_interval='30 0 * * *',
        tags=['my_dags'],
) as dag:
    python_task_context = PythonOperator(
        task_id='print_current_date_with_context_variable',
        python_callable=print_current_date_provide_context,
        provide_context=True  # True일 경우 Task Instance의 Attribute를 Keyword Argument로 받아서 사용할 수 있음
    )

    python_task_context
