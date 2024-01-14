from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from airflow.exceptions import AirflowFailException
from operators.slack_notifier import task_fail_slack_alert, task_succ_slack_alert


# slack_notifier 에 선언한 webhook 전송 함수를 활용하여 slack 알림을 제공합니다
default_args = {
    'owner': 'kyle',
    'depends_on_past': False,
    'start_date': datetime(2022, 4, 20),
    'retires': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': task_fail_slack_alert, # 실패 알림
    # 'on_success_callback': task_succ_slack_alert, # 성공 알림 필요 시 추가
}


def _handle_job_error() -> None:
    raise AirflowFailException("Raise Exception.")


with DAG(
        dag_id='python_dag_with_slack_webhook',
        default_args=default_args,
        schedule_interval='30 0 * * *',
        tags=['my_dags']
) as dag:
    execution_date = "{{ ds }}"

    send_slack_noti = PythonOperator(
        task_id='raise_exception_and_send_slack_noti',
        python_callable=_handle_job_error,
        op_args=[execution_date]
    )

    send_slack_noti
