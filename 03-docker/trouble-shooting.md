# 문제 해결 문서

이 문서는 Docker를 다루다 흔히 발생할 수 있는 문제들과 이를 해결할 수 있는 방법에 대해 다룹니다.
만약 문제를 겪고있다면, 이 문서에 내가 겪은 문제가 있나 찾아보세요.

## Docker를 설치했는데, 쉘에서 인식하지 못합니다.

### 문제 상황

Docker를 설치한 뒤, 터미널에서 `docker` 를 입력하면 다음과 같은 에러가 나옵니다.

```
zsh: command not found: docker
```

### 문제 원인

`command not found` 에러는 쉘에서 설치된 커맨드를 인식할 수 없는 경우에 발생합니다.
대부분의 원인은 다음과 같습니다.

- 실제로 command가 없는 경우
- system path에 command 가 없는 경우

이 경우 설치는 이미 하셨다고 하니, 후자일 가능성이 높습니다.

### 해결 방법

본인이 사용하고 있는 셸의 설정에 docker 실행 파일 경로를 `PATH` 환경 변수에 추가해주면 됩니다.

- Windows의 경우, PowerShell을 열어 다음 명령어를 쉘에서 입력합니다.
  - `[Environment]::SetEnvironmentVariable("Path", "$env:Path;C:\Program Files\Docker\Docker\resources\bin\docker.exe", "User")`
  - `C:\Program Files\Docker\Docker\resources\docker.exe` 이 실제로 존재하는지도 확인해보세요.
  - PowerShell을 재시작한 뒤, `docker` 명령어가 잘 인식하는지 확인해봅니다.
- Mac OS의 경우, 다음 명령어를 쉘에서 입력합니다.
  - `echo 'export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"' >> ~/.zshrc`
  - 만약 본인이 사용하는 쉘이 zsh이 아니라면(bash, fish 등), `.zshrc` 가 아닌 해당 쉘 설정 파일로 바꿔주세요.

## Docker 빌드 중 파일을 찾지 못합니다.

### 문제 상황

`docker build` 명령어로 컨테이너 이미지를 빌드하는데, 빌드 중 다음과 같은 에러가 나옵니다.

```
python: can't open file '/app/main.py': [Errno 2] No such file or directory
```

### 문제 원인

빌드 중 `No such file or directory` 에러는 말 그대로 빌드하는 컨테이너 내 파일 시스템에 해당 파일이나, 경로가 없어서 그렇습니다. 이 경우 '/app/main.py' 가 되겠군요.

대부분의 원인은 다음과 같습니다.

- `Dockerfile` 파일이 위치한 경로에서 `docker build` 명령어를 하지 않은 경우
- `Dockerfile` 내에서 `COPY` 등의 명령어를 누락하거나, 실제로 존재하지 않는 파일, 폴더 경로를 넘겨준 경우

전자의 경우, `Dockerfile` 이 위치한 경로로 쉘을 이동하신 뒤, `docker build` 를 해주면 해결됩니다.

여기서는 후자의 경우라고 생각하고 문제를 좀 더 살펴보겠습니다.

문제 상황에서 `Dockerfile` 은 다음과 같았습니다.

```Dockerfile
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

COPY . /app
WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN pip install pip==23.0.1 && \
    pip install poetry==1.2.1 && \
    poetry export -o requirements.txt && \
    pip install -r requirements.txt 

CMD ["python", "main.py"]
```

그리고 프로젝트 구조는 다음과 같았습니다.

```
app/main.py
Dockerfile
pyproject.toml
poetry.lock
```

빌드 중 에러가 난 부분은 `CMD ["python", "main.py"]` 이 부분입니다. 
뭐가 문제였을까요? 좀 더 살펴봅시다.

`CMD ["python", "main.py"]` 명령어 전까지 위 빌드를 진행하면 컨테이너 이미지 내 파일 시스템은 다음과 같은 파일과 디렉토리를 가지게 됩니다.

```
/app/app/main.py  # `COPY . /app` 의해 생성
/app/requirements.txt  # `poetry export -o requirements.txt` 에 의해 생성
```

그리고 `CMD ["python", "main.py"]` 실행 시, 쉘은  `WORKDIR /app` 에 의해 `/app` 경로에 있습니다.
이 상황에서 `python main.py` 를 실행하게 되는거죠.

하지만 위에서 보시다 시피, `app/` 경로 내에는 `app/main.py` 와 `requirements.txt` 만 존재합니다.
`main.py` 는 존재하지 않기 때문에 `No such file or directory` 에러가 발생합니다.

### 해결 방법

이 문제를 해결하는 방법은 2가지입니다.

첫 번째는 `Dockerfile` 내에 `CMD ["python", "main.py"]` 부분을 다음처럼 바꾸는 것입니다.

```
CMD ["python", "app/main.py"]
```

두 번째 방법은 프로젝트 구조를 수정하여 `app/main.py` 이 아니라 `main.py` 이 되게합니다.
즉 프로젝트 구조는 다음과 같아야 합니다.

```
main.py
Dockerfile
pyproject.toml
poetry.lock
```

## Docker 빌드 중 PEP 440 버전 에러가 납니다.

### 문제 상황

`docker build` 명령어로 컨테이너 이미지를 빌드하는데, 빌드 중 다음과 같은 에러가 나옵니다.

```
Invalid PEP 440 version
```

이 에러는 `Dockerfile` 내 다음 부분에서 발생했습니다.

```Dockerfile
RUN pip install pip==23.0.1 && \
    pip install poetry==1.2.1 && \
    pip install pip --upgrade && \
    poetry export -f requirements.txt && \
    pip install --user -r requirements.txt
```

### 문제 원인

이 에러는 pip 버전과 poetry 버전에 대한 호환성 문제입니다.

### 해결 방법

`Dockerfile` 내에서 `pip install pip --upgrade` 부분을 삭제해줍니다.
(pip를 이미 `23.0.1` 로 고정하여 install 한 상황에서 다시 버전 업데이트를 하는 것은 매우 어색합니다.)
