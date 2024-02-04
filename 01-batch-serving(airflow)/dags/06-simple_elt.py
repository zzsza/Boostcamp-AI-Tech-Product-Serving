# ELT 파이프라인을 개발합니다(Extract - Load - Transfer)
# 1) Extract : 데이터를 추출하는 과정. 여기선 Cloud Storage에 업로드(현업에선 Database에서 추출)
# 2) Load : Cloud Storage의 Bucket에 Data 저장된 것을 데이터 웨어하우스인 BigQuery로 저장
# 3) Transform : Load로 저장된 데이터를 Transform. BigQuery 쿼리로 데이터를 처리
# pip install apache-airflow-providers-google==10.14.0

from airflow import DAG
from datetime import datetime, timedelta
from pathlib import Path
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryExecuteQueryOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from utils.slack_notifier import task_fail_slack_alert, task_succ_slack_alert

execution_date = '{{ ds_nodash }}' #20240101
execution_date_with_slash = "{{ macros.ds_format(ds, '%Y-%m-%d', '%Y/%m/%d') }}" # 2024/01/01

# 아래 2개는 여러분들의 Google Cloud Project, Bucket 입력
PROJECT_ID = "boostcamp-ai-tech-serving"
BUCKET_NAME = "boostcamp-ai-tech-gcs"

FILE_NAME = f"bike_data_{execution_date}.csv"
LOCAL_FILE_PATH = str(Path(__file__).parent.parent / "data" / FILE_NAME) # 파일의 할아버지(dags -> 01-batch-serving(airflow))의 data 폴더

GCS_PATH = f"{execution_date_with_slash}/bike_data.csv"

default_args = {
    'owner': 'kyle',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'end_date': datetime(2024, 1, 4),
    'retires': 1,
    'retry_delay': timedelta(minutes=5),
}

schema_fields = [
  {
    "mode": "NULLABLE",
    "name": "trip_id",
    "type": "INTEGER"
  },
  {
    "mode": "NULLABLE",
    "name": "subscriber_type",
    "type": "STRING"
  },
  {
    "mode": "NULLABLE",
    "name": "bikeid",
    "type": "STRING"
  },
  {
    "mode": "NULLABLE",
    "name": "start_time",
    "type": "TIMESTAMP"
  },
  {
    "mode": "NULLABLE",
    "name": "start_station_id",
    "type": "INTEGER"
  },
  {
    "mode": "NULLABLE",
    "name": "start_station_name",
    "type": "STRING"
  },
  {
    "mode": "NULLABLE",
    "name": "end_station_id",
    "type": "STRING"
  },
  {
    "mode": "NULLABLE",
    "name": "end_station_name",
    "type": "STRING"
  },
  {
    "mode": "NULLABLE",
    "name": "duration_minutes",
    "type": "INTEGER"
  },
  {
    "mode": "NULLABLE",
    "name": "dummy_date",
    "type": "DATE"
  }
]


with DAG(
    dag_id='simple_elt_pipeline',
    default_args=default_args,
    schedule_interval='30 0 * * *',
    tags=['my_dags'],
    catchup=False,
    on_failure_callback=task_fail_slack_alert,
    # on_success_callback=task_succ_slack_alert  # 성공 알림 필요 시 추가
) as dag:

    extract_data = LocalFilesystemToGCSOperator(
        task_id="extract_data",
        src=LOCAL_FILE_PATH,
        bucket=BUCKET_NAME,
        dst=GCS_PATH
    )


    load_csv = GCSToBigQueryOperator(
        task_id="gcs_to_bigquery",
        gcp_conn_id="google_cloud_default",
        bucket=f'{BUCKET_NAME}', 
        source_objects=[GCS_PATH], 
        schema_fields=schema_fields,
        source_format='CSV',
        skip_leading_rows=1,
        create_disposition='CREATE_IF_NEEDED',
        destination_project_dataset_table=f'{PROJECT_ID}.temp.bike_{execution_date}', # temp dataset 생성 필요
        write_disposition='WRITE_TRUNCATE'
    )

    # dummy_date 별 COUNT
    sql_query = f"""
    SELECT 
      dummy_date, 
      start_station_id, 
      end_station_id, 
      COUNT(bikeid) as cnt
    FROM `{PROJECT_ID}.temp.bike_{execution_date}`
    GROUP BY 
      dummy_date, 
      start_station_id, 
      end_station_id
    """

    transform = BigQueryExecuteQueryOperator(
        task_id='run_query',
        sql=sql_query,
        use_legacy_sql=False,
        write_disposition='WRITE_TRUNCATE',
        destination_dataset_table=f"temp.bike_agg_{execution_date}"
    )

    extract_data >> load_csv >> transform