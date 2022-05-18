# Docker

## 도커 이미지 빌드
```shell
docker build -t <빌드할 이미지 이름> .
```

## 도커 이미지 실행
```shell
docker run [-p <접근할 포트>:<컨테이너 포트>] <이미지 이름>
```
자세한 옵션들은 [이 링크](https://docs.docker.com/engine/reference/run/) 를 참고해 주세요.

## 도커 컴포즈 실행
```shell
docker-compose up [-d]
```
