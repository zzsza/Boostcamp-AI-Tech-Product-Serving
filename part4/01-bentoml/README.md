# BentoML



## 환경 설정
```shell
poetry install
# or
$ pip install -r requirements.txt
```

## Frontend + Backend
```shell
bentoml serve MaskAPIService:latest
python -m streamlit run app/frontend.py
```