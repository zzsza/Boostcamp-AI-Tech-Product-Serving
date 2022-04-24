from airflow import DAG
from datetime import datetime, timedelta
from airflow.contrib.operators.gcs_to_bq import GoogleCloudStorageToBigQueryOperator
from airflow.contrib.operators.bigquery_operator import BigQueryOperator

# Google Cloud Provider 추가 설치가 필요합니다
  # pip3 install apache-airflow-providers-google
# Service Account JSON 파일 생성 후, google_cloud_default에 추가 필요
# Google Cloud Storage Bucket을 생성해야 합니다
# BigQuery에서 temp 데이터셋을 생성해야 합니다

PROJECT_ID = 'my-project-1541645429744' # TODO: 여러분들의 Google Cloud Platform Project ID 입력해주세요
BUCKET_NAME = 'kyle-school' # TODO: 여러분들의 BUCKET 이름을 넣어주세요

default_args = {
    'owner': 'kyle',
    'depends_on_past': False,
    'start_date': datetime(2022, 4, 20),
    'retries': 0,
    'retry_delay': timedelta(minutes=1),
    'end_date': datetime(2022, 4, 24),
    'project_id': PROJECT_ID
}

dag = DAG('simple_etl_storage_to_bigquery',
          default_args=default_args,
          schedule_interval='30 0 * * *') # UTC 기준 0시 30분(한국 시간 9시 30분)에 매일 실행

execution_date = '{{ ds_nodash }}'


storage_to_bigquery_task = GoogleCloudStorageToBigQueryOperator(
    dag=dag,
    google_cloud_storage_conn_id='google_cloud_default',
    bigquery_conn_id='google_cloud_default',
    task_id='storage_to_bigquery',
    schema_object='data/bike_schema.json',
    bucket=BUCKET_NAME, # 생성한 bucket 이름을 넣으세요
    source_objects=[f"data/bike_data_{execution_date}.csv"],
    source_format='CSV',
    destination_project_dataset_table=f'{PROJECT_ID}.temp.bike_{execution_date}',
    write_disposition='WRITE_TRUNCATE',
    skip_leading_rows=1
)

agg_query = f"""
SELECT 
  dummy_date, start_station_id, end_station_id, COUNT(bikeid) as cnt
FROM `{PROJECT_ID}.temp.bike_{execution_date}`
GROUP BY dummy_date, start_station_id, end_station_id
"""

query_task = BigQueryOperator(
        dag=dag,
        task_id="query_to_table",
        bigquery_conn_id='google_cloud_default',
        sql=agg_query,
        use_legacy_sql=False,
        write_disposition='WRITE_TRUNCATE',
        destination_dataset_table=f"temp.bike_agg_{execution_date}"
)

storage_to_bigquery_task >> query_task