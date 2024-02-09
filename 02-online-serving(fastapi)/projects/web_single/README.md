# Web Single Pattern ML API by FastAPI

Web Single Pattern ML API by FastAPI는 FastAPI를 이용해 만든 웹 서비스로, 단일 모델을 이용해 예측을 수행하는 API를 제공합니다.

## Pre-requisites

- Python >= 3.9
- Poetry >= 1.1.4

## Installation

```bash
poetry install
```

## Run

```bash
PYTHONPATH=.
poetry run python main.py
```

## Usage

### Predict

```bash
curl -X POST "http://0.0.0.0:8000/predict" -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

{"id":3,"result":0}
```

### Get all predictions

```bash
curl "http://0.0.0.0:8000/predict"

[{"id":1,"result":0},{"id":2,"result":0},{"id":3,"result":0}]
```

### Get a prediction

```bash
curl "http://0.0.0.0:8000/predict/1"
{"id":1,"result":0}
```

## Build

```bash
docker build -t web_single_example .
```

## Project Structure

```bash
.
├── .dockerignore    # 도커 이미지 빌드 시 제외할 파일 목록
├── .gitignore       # git에서 제외할 파일 목록
├── Dockerfile       # 도커 이미지 빌드 설정 파일
├── README.md        # 프로젝트 설명 파일
├── __init__.py
├── api.py           # API 엔드포인트 정의 파일
├── config.py        # Config 정의 파일
├── database.py      # 데이터베이스 연결 파일
├── db.sqlite3       # SQLite3 데이터베이스 파일
├── dependencies.py  # 앱 의존성 관련 로직 파일
├── main.py          # 앱 실행 파일
├── model.joblib     # 학습된 모델 파일
├── model.py         # 모델 관련 로직 파일
├── poetry.lock      # Poetry 라이브러리 버전 관리 파일
└── pyproject.toml   # Poetry 프로젝트 설정 파일
```