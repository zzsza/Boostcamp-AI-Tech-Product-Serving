# Airflow Docker

## 실행 방법
- Docker 이미지를 직접 실행하는 방법과 Docker Compose를 사용하는 방법 2가지를 다룹니다


### 1) Docker 이미지 직접 실행하는 방법

#### Docker Network 생성하기

Airflow의 모든 컴포넌트가 Docker 컨테이너로 배포될텐데, 이 컨테이너들간의 통신할 네트워크를 먼저 생성합니다.

다음 명령어로 Docker Network를 생성합니다.

```bash
$ docker network create airflow
```

잘 생성되었는지 다음처럼 확인해볼 수 있습니다.

```bash
$ docker network ls
NETWORK ID     NAME      DRIVER    SCOPE
b8f5eb31452e   airflow   bridge    local
```

#### Meta Database 사용하기
Meta Database로 PostgreSQL를 사용하겠습니다.

Postgres 컨테이너에 Volume 마운트할 디렉토리를 다음처럼 만들어둡니다.

```bash
$ mkdir data
```

다음 명령어로 Postgres 컨테이너를 실행합니다.

```bash
$ docker run \
  --name airflow-database \
  -d \
  --network airflow \
  -v $(pwd)/data:/var/lib/postgresql/data \
  -e POSTGRES_USER=airflow \
  -e POSTGRES_PASSWORD=1234 \
  postgres:13
```

컨테이너가 제대로 실행되었는지 다음처럼 확인할 수 있습니다.

```bash
$ docker ps

CONTAINER ID   IMAGE         COMMAND                  CREATED         STATUS              PORTS                    NAMES
c0b60f349279   postgres:13   "docker-entrypoint.s…"   3 minutes ago   Up About a minute   5432/tcp   airflow-database
```

#### Meta Database 초기화 하기

다음 명령어로 Meta Database를 초기화합니다.

```bash
$ docker run \
  --name airflow-init \
  --network airflow \
  --entrypoint /bin/bash \
  -e AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:1234@airflow-database:5432/airflow \
  apache/airflow:2.2.3-python3.8 \
  -c " \
    airflow db init && \
    airflow users create \
      --username admin \
      --password 1234 \
      --firstname heumsi \
      --lastname jeon \
      --role Admin \
      --email heumsi@naver.com \
  "
```

#### Scheduler 실행하기

다음 명령어로 Scheduler를 실행합니다

```bash
$ docker run \
  --name airflow-scheduler \
  --network airflow \
  -d \
  -e AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:1234@airflow-database:5432/airflow \
  -e AIRFLOW__CORE__EXECUTOR=LocalExecutor \
  -v $PWD/dags:/opt/airflow/dags \
  apache/airflow:2.2.3-python3.8 \
  airflow scheduler
```

컨테이너가 제대로 실행되었는지 다음처럼 확인할 수 있습니다.

```bash
$ docker ps

CONTAINER ID   IMAGE                            COMMAND                  CREATED          STATUS          PORTS      NAMES
f2ac94182d39   apache/airflow:2.2.3-python3.8   "/usr/bin/dumb-init …"   45 seconds ago   Up 44 seconds   8080/tcp   airflow-scheduler
...
```

#### Webserver 실행하기

다음 명령어로 Webserver를 실행합니다.

```bash
$ docker run \
  --name airflow-webserver \
  --network airflow \
  -d \
  -p 8080:8080 \
  -e AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:1234@airflow-database:5432/airflow \
  -v $PWD/dags:/opt/airflow/dags \
  apache/airflow:2.2.3-python3.8 \
  airflow webserver
```

```bash
$ docker ps

CONTAINER ID   IMAGE                            COMMAND                  CREATED          STATUS          PORTS      NAMES
e8dd306789f6   apache/airflow:2.2.3-python3.8   "/usr/bin/dumb-init …"   45 seconds ago   Up 41 seconds   0.0.0.0:8080->8080/tcp   airflow-webserver...
...
```

#### Code Server 실행하기

Code Server는 VSCode의 Web Browser 버전입니다.
서버에 직접 접속하여 DAG 파일을 작성하지 않고, 이 Code Server를 이용하여 작성할 수 있도록 해봅시다.

다음처럼 Docker 컨테이너로 실행합니다. 이 때 `dags/` 디렉토리를 마운트합니다.

```bash
$ docker run -it --name code-server \
    --name airflow-code-server \
    -d \
    -v "$(pwd)/dags:/home/coder/project" \
    -p 8888:8888 \
    -e PASSWORD=1234 \
    -e HOST=0.0.0.0 \
    -e PORT=8888 \
    codercom/code-server:4.0.2
```

컨테이너가 제대로 실행되었는지 다음처럼 확인할 수 있습니다.

```bash
$ docker ps

CONTAINER ID   IMAGE                         COMMAND                  CREATED          STATUS          PORTS                    NAMES
88608ae21d39   codercom/code-server:latest   "/usr/bin/entrypoint…"   7 seconds ago   Up 6 seconds   8080/tcp, 0.0.0.0:8888->8888/tcp   airflow-code-server
```

#### 실행한 Docker 컨테이너 확인

실행한 모든 컨테이너를 확인해보면 다음과 같습니다.

```bash
$ docker ps

CONTAINER ID   IMAGE                            COMMAND                  CREATED              STATUS              PORTS                              NAMES
e8dd306789f6   apache/airflow:2.2.3-python3.8   "/usr/bin/dumb-init …"   About a minute ago   Up About a minute   0.0.0.0:8080->8080/tcp             airflow-webserver
bb7e13d1f4c5   apache/airflow:2.2.3-python3.8   "/usr/bin/dumb-init …"   4 minutes ago        Up 4 minutes        8080/tcp                           airflow-scheduler
42736f3bf287   postgres:13                      "docker-entrypoint.s…"   5 minutes ago        Up 5 minutes        5432/tcp                           airflow-database
88608ae21d39   codercom/code-server:latest      "/usr/bin/entrypoint…"   7 seconds ago        Up 6 seconds        8080/tcp, 0.0.0.0:8888->8888/tcp   airflow-code-server
```


<br />

---

### 2) Docker Compose를 사용하는 방법
- [docker-compose.yml](https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/blob/main/part4/02-airflow/docker-compose.yml) 파일을 확인해주세요


아래 명령어로 Docker Compose를 실행합니다.

```bash
$ docker-compose up
```

실행한 모든 컨테이너를 확인해보면 다음과 같습니다.

```bash
$ docker-compose ps

      Name                     Command               State            Ports
------------------------------------------------------------------------------------
airflow-code-server   /usr/bin/entrypoint.sh --b ... Up       8080/tcp, 0.0.0.0:8888->8888/tcp
airflow-database    docker-entrypoint.sh postgres    Up       5432/tcp
airflow-init        /bin/bash -c  \                  Exit 0
                      airflow  ...
airflow-scheduler   /usr/bin/dumb-init -- /ent ...   Up       8080/tcp
airflow-webserver   /usr/bin/dumb-init -- /ent ...   Up       0.0.0.0:8080->8080/tcp
```


<br />

---



### Airflow Webserver 접속하기

브라우저에서 `http://localhost:8080` 에 접속하면 다음처럼 Airflow Webserver 화면을 볼 수 있습니다.

![airflow-webserver](./assets/images/airflow-webserver.png)

초기 계정 정보는 `admin` / `1234` 입니다.

### Code Server 접속하기

브라우저에서 `http://localhost:8888` 에 접속하면 다음처럼 Code Server 화면을 볼 수 있습니다.

![code-server](./assets/images/code-server.png)

초기 비밀번호는 `1234` 입니다.

`dags/` 폴더에서 Airflow DAG 파일을 생성하면 Airflow Webserver에서 확인할 수 있습니다

만약 Code Server를 사용하지 않는다면, `$AIRFLOW_HOME/dags`에 파일을 옮기면 Airflow Scheduler가 파일을 파싱해 Airflow Webserver에서 확인할 수 있습니다
