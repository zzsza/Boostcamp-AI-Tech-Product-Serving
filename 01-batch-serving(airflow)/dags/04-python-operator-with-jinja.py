from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# 앞의 03-python-operator-with-context는 provide_context=True 옵션을 주고 Attribute에 접근
# 이 방식이 아닌 Airflow의 Template 방식을 사용. Jinja Template => Flask 자주 활용되는 템플릿
# Python에서는 Template이랑 provide_context=True와 큰 차이를 못 느낄 수도 있지만, SQL Opearator나 다른 오퍼레이터에선 유용하게 사용됨(템플릿)
# 쿼리문(WHERE절)에 Airflow의 execution_date를 인자로 넣고 실행
# Jinja Template : Airflow의 미리 정의된 템플릿. {{ ds }}, {{ ds_nodash }} 라고 정의
# Airflow Operator에 넘겨주면 실행 과정에서 템플릿 기반으로 값이 업데이트됨


default_args = {
    'owner': 'kyle',
    'depends_on_past': False,  # 이전 DAG의 Task가 성공, 실패 여부에 따라 현재 DAG 실행 여부가 결정. False는 과거의 실행 결과 상관없이 매일 실행한다
    'start_date': datetime(2022, 4, 20),
    'retires': 1,  # 실패시 재시도 횟수
    'retry_delay': timedelta(minutes=5)  # 만약 실패하면 5분 뒤 재실행
}


# 사용할 함수 정의
def print_current_date_with_jinja(date):
    execution_date = datetime.strptime(date, "%Y-%m-%d").date()
    date_kor = ["월", "화", "수", "목", "금", "토", "일"]
    datetime_weeknum = execution_date.weekday()
    print(f"{execution_date}는 {date_kor[datetime_weeknum]}요일입니다")


# with 구문으로 DAG 정의
with DAG(
        dag_id='python_dag_with_jinja',
        default_args=default_args,
        schedule_interval='30 0 * * *',
        tags=['my_dags']
) as dag:
    execution_date = "{{ ds }}"  # Template 정의

    python_task_jinja = PythonOperator(
        task_id='print_current_date_with_jinja',
        python_callable=print_current_date_with_jinja,
        op_args=[execution_date]
    )

    python_task_jinja
