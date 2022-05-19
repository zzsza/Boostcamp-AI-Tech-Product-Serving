from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# 앞선 02-python-operator.py는 "date_now = datetime.now().date()"를 사용하기 때문에
# 언제 실행해도 실행하는 시간 기준으로 실행됨
# Airflow는 Batch 성으로 특정 시간대로 실행하는 도구인데, 위와 같이 now 등을 사용하면 실행하는 시점 기준으로 실행이 됩니다(원래 기대했던 실행 시점이 아닌, 동작 시점)
# Airflow는 항상 현재 최신 작업만 실행하는 것은 아니고, 과거 날짜를 실행해야 하는 경우도 있음(Backfill이란 용어 사용)
# 따라서 코드 상에서 now(), SQL 상에서 current_date() 등을 사용하지 않고, Airflow에서 실행하기로 했던 시간을 넣어줘야 합니다
# execution_date라고 부름

default_args = {
    'owner': 'kyle',
    'depends_on_past': False,  # 이전 DAG의 Task가 성공, 실패 여부에 따라 현재 DAG 실행 여부가 결정. False는 과거의 실행 결과 상관없이 매일 실행한다
    'start_date': datetime(2022, 4, 20),
    'retires': 1,  # 실패시 재시도 횟수
    'retry_delay': timedelta(minutes=5)  # 만약 실패하면 5분 뒤 재실행
}


def print_current_date_with_context_variable(*args, **kwargs):
    """
    {'conf': <airflow.configuration.AirflowConfigParser object at 0x1069e8b00>,
    'dag': <DAG: python_dag_with_context>,
    'dag_run': <DagRun python_dag_with_context @ 2022-04-20 00:30:00+00:00: scheduled__2022-04-20T00:30:00+00:00,
    externally triggered: False>,
    'data_interval_end': DateTime(2022, 4, 21, 0, 30, 0, tzinfo=Timezone('UTC')),
    'data_interval_start': DateTime(2022, 4, 20, 0, 30, 0, tzinfo=Timezone('UTC')),
    'ds': '2022-04-20',
    'ds_nodash': '20220420',
    'next_execution_date': <Proxy at 0x108658248 with factory <function TaskInstance.get_template_context.<locals>.deprecated_proxy.<locals>.deprecated_func at 0x108654b70>>
    }
    """
    print(f"kwargs : {kwargs}")
    execution_date = kwargs['ds']
    execution_date_nodash = kwargs['ds_nodash']
    print(f"execution_date_nodash : {execution_date_nodash}")
    execution_date = datetime.strptime(execution_date, "%Y-%m-%d").date()
    date_kor = ["월", "화", "수", "목", "금", "토", "일"]
    datetime_weeknum = execution_date.weekday()
    print(f"{execution_date}는 {date_kor[datetime_weeknum]}요일입니다")


# with 구문으로 DAG 정의
with DAG(
        dag_id='python_dag_with_context',
        default_args=default_args,
        schedule_interval='30 0 * * *',
        tags=['my_dags']
) as dag:
    PythonOperator(
        task_id='print_current_date_with_context_variable',
        python_callable=print_current_date_with_context_variable,
        provide_context=True  # True일 경우에 Airflow Task Instance의 Attribute를 Keyword Argument로 Python 함수에서 사용할 수 있음
    )

    # task가 1개일 경우엔 순서를 명시하지 않아도 실행
