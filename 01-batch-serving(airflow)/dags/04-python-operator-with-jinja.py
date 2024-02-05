from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# 앞의 03-python-operator-with-context는 kwargs로 여러 정보를 같이 주입. ds, ds_nodash
# Jinja Template 사용하면 ds를 kwargs['ds']. {{ ds }}
# Flask, FastAPI에서도 자주 사용
# Python에서는 kwargs로 접근하시면 빠르게 가능. SQL. 쿼리문에서 WHERE 조건에 exeuction_date="2024-01-01"
# {{ ds }}, {{ ds_nodash }}
# Airflow의 Operator에 template_fields 여기에 있는 값들은 Airflow가 실행 과정에서 {{ ds }} => "2024-01-01" 변환

default_args = {
    "owner": "kyle",
    "depends_on_past": False, # 이전 DAG의 Task 성공 여부에 따라서 현재 Task를 실행할지 말지가 결정. False는 과거 Task의 성공 여부와 상관없이 실행
    "start_date": datetime(2024, 1, 1),
    "end_date": datetime(2024, 1, 4)
}


def print_current_date_with_jinja(date):
    execution_date = datetime.strptime(date, "%Y-%m-%d").date()
    date_kor = ["월", "화", "수", "목", "금", "토", "일"]
    datetime_weeknum = execution_date.weekday()
    print(f"{execution_date}는 {date_kor[datetime_weeknum]}요일입니다")

    return execution_date

with DAG(
    dag_id="python_dag_with_jinja",
    default_args=default_args,
    schedule_interval="30 0 * * *",
    tags=['my_dags'],
    catchup=True
) as dag:
    execution_date = "{{ ds }}"

    python_task_jinja = PythonOperator(
        task_id="print_current_date_with_jinja",
        python_callable=print_current_date_with_jinja,
        # op_args=[execution_date]
        op_kwargs = {
            "date": execution_date
        }
    )

    python_task_jinja