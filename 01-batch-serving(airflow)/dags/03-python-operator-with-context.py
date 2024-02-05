from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# 앞선 02-python-operator.py는 "date_now = datetime.now().date()"를 사용했기 때문에
# 언제 실행해도 우리가 실행하는 시간 기준으로 실행됨
# Airflow Batch성으로 특정 시간대로 실행하는 도구. now 등을 잘 쓰지 않음. 의도한 시간, 날짜 주입해서 사용
# Airflow로 과거 날짜로 실행해야 하는 경우도 존재. 과거 데이터 마이그레이션
# 코드 상에서 now(), SQL 상에서도 current_date() 사용하지 않고, Airflow에서 실행하기로 했던 시간을 넣어줘야 함
# execution_date, logical_date
# 멱등성 : 연산을 여러 번 적용하더라도 결과가 달라지지 않는 성질

default_args = {
    "owner": "kyle",
    "depends_on_past": False, # 이전 DAG의 Task 성공 여부에 따라서 현재 Task를 실행할지 말지가 결정. False는 과거 Task의 성공 여부와 상관없이 실행
    "start_date": datetime(2024, 1, 1),
    "end_date": datetime(2024, 1, 4)
}

def print_current_date_with_context(*args, **kwargs):
    """
    kwargs: {'conf': <airflow.configuration.AirflowConfigParser object at 0x1037d9730>, 
    'dag': <DAG: python_dag_with_context>, 
    'dag_run': <DagRun python_dag_with_context @ 2024-01-01 00:30:00+00:00: scheduled__2024-01-01T00:30:00+00:00, 
    state:running, 
    queued_at: 2024-02-05 04:37:43.715262+00:00. externally triggered: False>, 
    'data_interval_end': DateTime(2024, 1, 2, 0, 30, 0, tzinfo=Timezone('UTC')), 'data_interval_start': DateTime(2024, 1, 1, 0, 30, 0, tzinfo=Timezone('UTC')), 
    'ds': '2024-01-01', 'ds_nodash': '20240101', 'execution_date': <Proxy at 0x110ad8440 with factory functools.partial(<function lazy_mapping_from_context.<locals>._deprecated_proxy_factory at 0x110a40670>, 'execution_date', DateTime(2024, 1, 1, 0, 30, 0, tzinfo=Timezone('UTC')))>, 'expanded_ti_count': None, 'inlets': [], 'logical_date': DateTime(2024, 1, 1, 0, 30, 0, tzinfo=Timezone('UTC')), 'macros': <module 'airflow.macros' from '/Users/kyle/boostcamp-ai-tech/.venv/lib/python3.9/site-packages/airflow/macros/__init__.py'>, 'next_ds': <Proxy at 0x110ade500 with factory functools.partial(<function lazy_mapping_from_context.<locals>._deprecated_proxy_factory at 0x110a40670>, 'next_ds', '2024-01-02')>, 'next_ds_nodash': <Proxy at 0x110ae5240 with factory functools.partial(<function lazy_mapping_from_context.<locals>._deprecated_proxy_factory at 0x110a40670>, 'next_ds_nodash', '20240102')>, 'next_execution_date': <Proxy at 0x110ae5500 with factory functools.partial(<function lazy_mapping_from_context.<locals>._deprecated_proxy_factory at 0x110a40670>, 'next_execution_date', DateTime(2024, 1, 2, 0, 30, 0, tzinfo=Timezone('UTC')))>, 'outlets': [], 'params': {}, 'prev_data_interval_start_success': None, 'prev_data_interval_end_success': None, 'prev_ds': <Proxy at 0x1109393c0 with factory functools.partial(<function lazy_mapping_from_context.<locals>._deprecated_proxy_factory at 0x110a40670>, 'prev_ds', '2023-12-31')>, 'prev_ds_nodash': <Proxy at 0x110ae8040 with factory functools.partial(<function lazy_mapping_from_context.<locals>._deprecated_proxy_factory at 0x110a40670>, 'prev_ds_nodash', '20231231')>, 'prev_execution_date': <Proxy at 0x110ae80c0 with factory functools.partial(<function lazy_mapping_from_context.<locals>._deprecated_proxy_factory at 0x110a40670>, 'prev_execution_date', DateTime(2023, 12, 31, 0, 30, 0, tzinfo=Timezone('UTC')))>, 'prev_execution_date_success': <Proxy at 0x110ae8100 with factory functools.partial(<function lazy_mapping_from_context.<locals>._deprecated_proxy_factory at 0x110a40670>, 'prev_execution_date_success', None)>, 'prev_start_date_success': None, 'run_id': 'scheduled__2024-01-01T00:30:00+00:00', 'task': <Task(PythonOperator): print_current_date_with_context>, 'task_instance': <TaskInstance: python_dag_with_context.print_current_date_with_context scheduled__2024-01-01T00:30:00+00:00 [running]>, 'task_instance_key_str': 'python_dag_with_context__print_current_date_with_context__20240101', 'test_mode': False, 'ti': <TaskInstance: python_dag_with_context.print_current_date_with_context scheduled__2024-01-01T00:30:00+00:00 [running]>, 'tomorrow_ds': <Proxy at 0x110ae8240 with factory functools.partial(<function lazy_mapping_from_context.<locals>._deprecated_proxy_factory at 0x110a40670>, 'tomorrow_ds', '2024-01-02')>, 'tomorrow_ds_nodash': <Proxy at 0x110ae8300 with factory functools.partial(<function lazy_mapping_from_context.<locals>._deprecated_proxy_factory at 0x110a40670>, 'tomorrow_ds_nodash', '20240102')>, 'triggering_dataset_events': <Proxy at 0x110a95a80 with factory <function TaskInstance.get_template_context.<locals>.get_triggering_events at 0x110a8fca0>>, 'ts': '2024-01-01T00:30:00+00:00', 'ts_nodash': '20240101T003000', 'ts_nodash_with_tz': '20240101T003000+0000', 'var': {'json': None, 'value': None}, 'conn': None, 'yesterday_ds': <Proxy at 0x110ae8400 with factory functools.partial(<function lazy_mapping_from_context.<locals>._deprecated_proxy_factory at 0x110a40670>, 'yesterday_ds', '2023-12-31')>, 'yesterday_ds_nodash': <Proxy at 0x110ae8480 with factory functools.partial(<function lazy_mapping_from_context.<locals>._deprecated_proxy_factory at 0x110a40670>, 'yesterday_ds_nodash', '20231231')>, 'templates_dict': None}
    """
    print(f"kwargs: {kwargs}")
    execution_date = kwargs["ds"]
    execution_date_nodash = kwargs["ds_nodash"]
    print(f"execution_date_nodash : {execution_date_nodash}")
    execution_date = datetime.strptime(execution_date, "%Y-%m-%d").date()
    date_kor = ["월", "화", "수", "목", "금", "토", "일"]
    datetime_weeknum = execution_date.weekday()
    print(f"{execution_date}는 {date_kor[datetime_weeknum]}요일입니다")



with DAG(
    dag_id="python_dag_with_context",
    default_args=default_args,
    schedule_interval="30 0 * * *",
    tags=['my_dags'],
    catchup=True
) as dag:
    
    PythonOperator(
        task_id="print_current_date_with_context",
        python_callable=print_current_date_with_context
    )