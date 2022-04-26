from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# 앞의 03 예제는 provide_context=True 옵션을 주고 Attribute에 접근함
# 이번엔 이 방식이 아닌 Airflow의 Template 방식을 사용
# Python에선 큰 차이를 못 느낄 수 있지만, SQL Operator 등에선 유용하게 사용됨(쿼리문에 Airflow Template을 사용)
# Jinja Template : Airflow에 미리 정의된 템플릿으로 {{ ds }}, {{ ds_nodash }} 라고 정의해서 Airflow Operator에 넘겨주면 실행하는 과정에서 템플릿 기반으로 값이 변환됨

default_args = {
                   'owner': 'kyle',
                   'depends_on_past': False,
                   'start_date': datetime(2022, 4, 20),
                   'retries': 1,
                   'retry_delay': timedelta(minutes=5),
               },


# 사용할 함수 정의
def print_current_date_jinja(execution_date):
    execution_date = datetime.strptime(execution_date, "%Y-%m-%d").date()
    date_kor = ["월", "화", "수", "목", "금", "토", "일"]
    datetime_weeknum = execution_date.weekday()
    print(f"{execution_date}는 {date_kor[datetime_weeknum]}요일입니다")


# with 구문으로 DAG 정의
with DAG(
        dag_id='python_dag_with_jinja',
        default_args=default_args,
        schedule_interval='30 0 * * *',
        tags=['my_dags'],
) as dag:
    execution_date = "{{ ds }}"  # Template 정의

    python_task_jinja = PythonOperator(
        task_id='print_current_date_with_jinja',
        python_callable=print_current_date_jinja,
        op_args=[execution_date],  # Argument 주입
        dag=dag,
    )

    python_task_jinja
