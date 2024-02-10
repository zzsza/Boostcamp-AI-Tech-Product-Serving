# Airflow Docker
- [Docker Install](https://docs.docker.com/engine/install/)을 참고해 Docker Engine을 설치합니다
- 만약 윈도우를 사용하고 있다면, WSL을 설치한 후, [WSL 2에서 Docker 원격 컨테이너 시작](https://learn.microsoft.com/ko-kr/windows/wsl/tutorials/wsl-containers) 글을 참고해 환경 설정을 해주세요


## Docker Compose를 사용하는 방법
- [docker-compose.yml](https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/blob/main/01-batch-serving(airflow)/docker-compose.yml) 파일을 확인해주세요

아래 명령어로 Docker Compose를 실행합니다. (처음엔 `-d` 옵션을 주지 않고 실행해보고, 정상적으로 확인된다면 그 후에 `-d` 옵션을 사용하는 것을 추천합니다)

```bash
cd boostcamp-ai-tech-product-serving/01-batch-serving(airflow)
docker-compose up -d
```

실행한 모든 컨테이너를 확인하고 싶다면 다음 명령어를 실행합니다.

```bash
docker-compose ps
```

- Localhost:8080으로 접속한 후, Airflow 화면이 보이는지 확인합니다

