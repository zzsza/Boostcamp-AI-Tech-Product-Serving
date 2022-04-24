from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'kyle',
    'depends_on_past': False,
    'start_date': datetime(2022, 4, 20),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG('bash_dag',
          default_args=default_args,
          schedule_interval='@once')

task1 = BashOperator(
    task_id='print_date',
    bash_command='date',
    dag=dag)

task2 = BashOperator(
    task_id='sleep',
    bash_command='sleep 5',
    retries=2,
    dag=dag)

task3 = BashOperator(
    task_id='pwd',
    bash_command='pwd',
    dag=dag)


task1 >> task2
task1 >> task3