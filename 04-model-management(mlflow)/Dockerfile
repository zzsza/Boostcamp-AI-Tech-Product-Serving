FROM python:3.9.10-slim-buster

COPY requirements.txt mlflow/requirements.txt
WORKDIR mlflow

RUN apt-get update && \
    pip install pip --upgrade && \
    pip install -r requirements.txt

CMD ["mlflow", "ui", "-h", "0.0.0.0", "-p", "5000"]
