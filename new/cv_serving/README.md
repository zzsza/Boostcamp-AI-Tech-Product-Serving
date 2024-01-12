# CV Serving Example through Web Single Pattern

CV 모델을 Web Single Pattern으로 서빙하는 예제입니다.

## Pre-requisites

- Python >= 3.9
- Poetry >= 1.1.4

## Installation

```bash
poetry install
```

## Run

먼저 `.env` 파일을 생성하고, 아래와 같이 모델 경로를 설정합니다.

```bash
# .env
MODEL_PATH=./model/model.pkl
```

그리고 아래와 같이 실행합니다.

```bash
PYTHONPATH=.
poetry run python main.py
```

## Usage

### Predict

```bash
curl -X POST -F "file=@./example_images/0a101263343a4a60a8dcd94d1fc8e8e253dadf14.jpg" http://0.0.0.0:8000/predict
 
{"id":1,"result":4}%
```

### Get all predictions

```bash
curl "http://0.0.0.0:8000/predict"

[{"id":1,"result":0},{"id":2,"result":4},{"id":3,"result":3}]
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
├── dependencies.py  # 앱 의존성 관련 로직 파일
├── main.py          # 앱 실행 파일
├── model/           # 모델 관련 디렉토리
├── poetry.lock      # Poetry 라이브러리 버전 관리 파일
└── pyproject.toml   # Poetry 프로젝트 설정 파일
```