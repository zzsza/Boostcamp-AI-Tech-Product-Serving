# MLflow

## 실행 방법

### Docker

먼저 아래 명령어로 Docker 이미지를 빌드합니다.

```bash
$ docker build -t mlflow:1.24.0 .
```

빌드되면 다음처럼 확인할 수 있습니다.

```bash
$ docker images
mlflow                                    1.24.0            072b9199e5d1   2 minutes ago   611MB
```

이제 이미지를 컨테이너로 실행합니다.

```bash
$ docker run --name mlflow -p 5000:5000 -v $(pwd):/mlflow --rm mlflow:1.24.0

[2022-04-09 06:28:13 +0000] [12] [INFO] Starting gunicorn 20.1.0
[2022-04-09 06:28:13 +0000] [12] [INFO] Listening at: http://0.0.0.0:5000 (12)
[2022-04-09 06:28:13 +0000] [12] [INFO] Using worker: sync
[2022-04-09 06:28:13 +0000] [13] [INFO] Booting worker with pid: 13
```

### Docker-compose

아래 명령어로 Docker Compose를 실행합니다.

```bash
$ docker-compose up

Docker Compose is now in the Docker CLI, try `docker compose up`

Creating network "04-mlflow_default" with the default driver
Creating 04-mlflow_web-ui_1 ... done
Attaching to 04-mlflow_web-ui_1
web_ui_1  | [2022-04-09 06:39:22 +0000] [13] [INFO] Starting gunicorn 20.1.0
web_ui_1  | [2022-04-09 06:39:22 +0000] [13] [INFO] Listening at: http://0.0.0.0:5000 (13)
web_ui_1  | [2022-04-09 06:39:22 +0000] [13] [INFO] Using worker: sync
web_ui_1  | [2022-04-09 06:39:22 +0000] [14] [INFO] Booting worker with pid: 14
```

## 사용 방법

브라우저에서 `http://0.0.0.0:5000/` 에 접속하면 다음처럼 MLflow Web UI를 확인할 수 있습니다.

![web ui](./assets/images/mlflow-web-ui.png)
