# Docker

## 도커 이미지 빌드
```shell
docker build -t <빌드할 이미지 이름:태그 이름> .
# ex. docker build -t 02-docker:latest .
```

## 도커 이미지 실행
```shell
docker run [-p <접근할 포트>:<컨테이너 포트>] [-v <로컬 마운트 경로>:<컨테이너 마운트 경로>] <이미지 이름>
# ex.  docker run -v $(pwd):/app 02-docker:latest
```
자세한 옵션들은 [이 링크](https://docs.docker.com/engine/reference/run/) 를 참고해 주세요.

## 도커 컴포즈 실행
```shell
docker-compose up [-d]
```

## 더 읽어보면 좋을 아티클들

도커와 관련되어 [더 읽어보면 좋을 내용들](./futher-readings.md)을 따로 공유해둡니다.

## 문제가 생겼을 경우

문제가 생긴 경우, [문제 해결 문서](./trouble-shooting.md)에 내가 겪은 상황이 있는지 확인해보세요.
