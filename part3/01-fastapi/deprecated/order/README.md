# FastAPI
FastAPI을 이용한 모델 온라인 서빙 

## Getting Started
0. Python requirements  
   `Python`: 3.6.2 이상이 필요합니다  
   `가상환경`: poetry(>=1.0.0)를 사용(권장), 또는 virtualenv, pyenv-virtualenv 등의 방법을 사용할 수 있습니다.
1. Installation
   1. 가상 환경을 설정합니다
      - Poetry
         1. Poetry 설치하기
            - Window
              ```shell
               # powershell
              > (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
              ```
            - Mac OSX / Linux
              ```shell
              > curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
              ```
         2. Poetry shell  
            Poetry로 가상환경을 만듭니다. 
            ```shell
            # 프로젝트 루트에서
            > cd $PROJECT_ROOT
            # (optional)
            > poetry config virtualenvs.create true # 로컬에 .venv 폴더를 생성해서, IDE에서 interpreter 설정이 편해집니다
            > poetry shell
            ```
      - virtualenv
        ```shell
        > python3 -m virtualenv venv
        > source $venv_name/bin/activate
        ```
      - pyenv-virtualenv
        ```shell
        > pyenv virtualenv $PYTHON_VERSION $VENV_NAME
        > pyenv shell $VENV_NAME
        ```
   2. 프로젝트의 의존성을 설치합니다
      - Using Poetry
        ```shell
        > poetry install
        ```
      - 나머지
        ```shell
        > pip install -r requirements.txt 
        ``` 
   3. 애플리케이션을 실행합니다
      ```shell
      > cd $part3/01-fastapi
      > python -m app
       INFO:     Started server process [11467]
       INFO:     Waiting for application startup.
       INFO:     Application startup complete.
       INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)

      ```  
   4. Frontend(Streamlit)와 Server를 같이 실행합니다
      ```shell
      make -j 2 run_app
      # or
      
      python3 -m app
      # in other shell
      python3 -m streamlit run app/frontend.py
      ```


## Reference
- [나의 파이썬 환경 구축기 2 - pyenv + poetry](https://dailyheumsi.tistory.com/244?category=799302)