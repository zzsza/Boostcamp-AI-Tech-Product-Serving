# RecSys Serving Example by FastAPI
RecSys 모델을 Web Single Pattern 으로 구현하기 위해 FastAPI 를 사용합니다.

## Pre-requisites

- Python >= 3.9
- Poetry >= 1.1.4

## Installation

```bash
poetry install
```

## Run
환경 설정
```bash
# .env
# Default Baseline for DeepCoNN
MODEL_PATH=./data/src/model_versions/model.pt
PYTHONPATH=
```
실행
```bash
poetry run python main.py
```

## Usage

### Train & Recommendation

```bash
# train 값 설정을 통해 학습 유무를 결정할 수 있습니다
# Default model.pt 은 DeepCoNN 모델에 대한 파일입니다.
curl -X POST "http://0.0.0.0:8000/scoring" -H "Content-Type: application/json" -d '{"model": "DeepCoNN", "train": true, "vector_create": true}'
{"id":11676,"isbn": "0000000000", "rating": 0.0, "model":"FM"} # example
```

### Get a Predictions
```bash
# 전체 결과 검색
curl -X GET "http://0.0.0.0:8000/scoring"
# 특정 결과 검색
curl -X GET "http://0.0.0.0:8000/scoring/{user_id}"
```

## Build

```bash
docker build -t web_single_example .
```

## Project Structure

```bash
.
├── /data                   # Data 관련 파일 및 코드
│   └── /src                # input 파일 저장 경로
│       ├── /images         # image input 파일 저장 경로
│       ├── /model_versions # 모델 weight 저장
│       ├── /submit         # 제출 파일(submission.csv) 저장 경로
│       └── /text_vector    # 벡터 파일
├── /log                    # 로그 파일 저장 경로
├── /models                 # 모델 구현 코드
├── .dockerignore           # 도커 이미지 빌드 시 제외할 파일 목록
├── .gitignore              # git에서 제외할 파일 목록
├── Dockerfile              # 도커 이미지 빌드 설정 파일
├── README_nlp.md           # 프로젝트 설명 파일
├── __init__.py
├── api.py                  # API 엔드포인트 정의 파일
├── config.py               # Config 정의 파일
├── database.py             # 데이터베이스 연결 파일
├── db.sqlite3              # SQLite3 데이터베이스 파일
├── dependencies.py         # 앱 의존성 관련 로직 파일
├── main.py                 # 앱 실행 파일
├── model.py                # 모델 관련 로직 파일
├── poetry.lock             # Poetry 라이브러리 버전 관리 파일
└── pyproject.toml          # Poetry 프로젝트 설정 파일
```