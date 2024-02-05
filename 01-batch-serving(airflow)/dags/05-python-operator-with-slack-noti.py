# slack_notifier에 선언한 webhook 전송 함수를 활용해 slack 알림을 제공하는 예제

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from airflow.exceptions import AirflowFailException
from utils.slack_notifier import task_fail_slack_alert, task_succ_slack_alert

default_args = {
    "owner": "kyle",
    "depends_on_past": False, # 이전 DAG의 Task 성공 여부에 따라서 현재 Task를 실행할지 말지가 결정. False는 과거 Task의 성공 여부와 상관없이 실행
    "start_date": datetime(2024, 1, 1),
    "end_date": datetime(2024, 1, 4)
}

def _handle_job_error() -> None:
    raise AirflowFailException("Raise Exception.")

with DAG(
    dag_id="python_dag_with_slack_webhook",
    default_args=default_args,
    schedule_interval="30 0 * * * ",
    tags=["my_dags"],
    catchup=True,
    on_failure_callback=task_fail_slack_alert
) as dag:
    
    send_slack_noti = PythonOperator(
        task_id="raise_exception_and_send_slack_noti",
        python_callable=_handle_job_error
    )

    send_slack_noti
