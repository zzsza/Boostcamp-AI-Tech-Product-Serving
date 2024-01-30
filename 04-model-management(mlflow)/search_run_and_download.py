from mlflow import MlflowClient, artifacts
from mlflow.entities import ViewType

query = "params.l1_ratio = '0.03' and metrics.`training_score` >= 0.65"

run = MlflowClient().search_runs(
    experiment_ids="961060126594144107", # my-first-experiment의 Experiment ID
    # filter_string='', # 아무 조건을 주고 싶지 않다면
    filter_string=query,
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=1,
    order_by=["metrics.training_score DESC"],
)[0]

# print("Run Info", run)

def download_model(run_id, model_name="model"):
    print("Download Model")
    artifact_uri=f"runs:/{run_id}/{model_name}"
    artifacts.download_artifacts(artifact_uri, dst_path=".")
    print("Download Model Finish")

download_model(run_id=run.info.run_id)
