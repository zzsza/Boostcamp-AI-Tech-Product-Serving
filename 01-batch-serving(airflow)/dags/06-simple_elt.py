# ELT 파이프라인을 개발합니다(Extract - Load - Trasnfer). 데이터 엔지니어링을 진행할 때, ELT / ETL
# 1) Extract : 데이터를 추출하는 과정. 여기선 Cloud Storage에 업로드(현업에선 Database에서 추출)
# 2) Load : Cloud Stroage의 Bucket에 Data가 저장. 저장된 Data를 데이터 웨어하우스인 BigQuery로 저장
# 3) Transform : Load로 저장된 데이터를 Transform. BigQuery 쿼리로 데이터를 처리
# pip install apache-airflow-providers-google

from airflow import DAG
from datetime import datetime, timedelta
from pathlib import Path
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator # GCS 데이터를 BigQuery로 옮김
from airflow.providers.google.cloud.operators.bigquery import BigQueryExecuteQueryOperator # BigQuery에서 Query를 실행
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator # Local에서 GCS로 데이터를 옮김
from utils.slack_notifier import task_fail_slack_alert, task_succ_slack_alert

execution_date = "{{ ds_nodash }}" # 20240101

PROJECT_ID = "boostcamp-ai-tech-serving"
BUCKET_NAME = "boostcamp-ai-tech-gcs"

FILE_NAME = f"bike_data_{execution_date}.csv" # bike_data_20240101.csv
LOCAL_FILE_PATH = str(Path(__file__).parent.parent / "data" / FILE_NAME)

GCS_PATH = f"{execution_date}/bike_data.csv" 
# 현업에서 데이터를 저장할 때, 날짜로 구분을 많이 함
# 2024/01/01/bike_data.csv
# 2024/01/02/bike_data.csv

default_args = {
    "owner": "kyle",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "end_date": datetime(2024, 1, 4)
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
    dag_id="simpel_elt_pipeline",
    default_args=default_args,
    schedule_interval="30 0 * * *",
    tags=["my_dags"],
    catchup=True
) as dag:
    
    # 1) Extract : Local To GCS
    extract_data = LocalFilesystemToGCSOperator(
        task_id="extract_data",
        src=LOCAL_FILE_PATH,
        dst=GCS_PATH,
        bucket=BUCKET_NAME
    )
    
    # 2) Load : GCS To BigQuery
    load_csv = GCSToBigQueryOperator(
        task_id="gcs_to_bigquery",
        bucket=BUCKET_NAME,
        source_objects=[GCS_PATH],
        destination_project_dataset_table=f"{PROJECT_ID}.temp.bike_{execution_date}", # temp.bike_20240101
        schema_fields=schema_fields,
        source_format='CSV',
        skip_leading_rows=1,
        create_disposition="CREATE_IF_NEEDED",
        write_disposition="WRITE_TRUNCATE",
        location="US"
    )

    # 3) Transform : BigQuery에서 Query 실행해서 다시 BigQuery에 저장
    # dummy_date 별 COUNT
    sql_query = f"""
    SELECT
      dummy_date,
      start_station_id,
      end_station_id,
      COUNT(bikeid) AS cnt
    FROM `{PROJECT_ID}.temp.bike_{execution_date}`
    GROUP BY
      dummy_date,
      start_station_id,
      end_station_id
    """
    transform = BigQueryExecuteQueryOperator(
        task_id="run_query",
        sql=sql_query,
        use_legacy_sql=False,
        allow_large_results=True,
        write_disposition="WRITE_TRUNCATE",
        destination_dataset_table=f"{PROJECT_ID}.temp.bike_agg"
    )

    extract_data >> load_csv >> transform