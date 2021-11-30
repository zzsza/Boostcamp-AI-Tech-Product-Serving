# Boostcamp-AI-Tech-Product-Serving
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

- [부스트캠프 AI Tech](https://boostcamp.connect.or.kr/program_ai.html) - Product Serving 자료


<br />

## Repository 구조
- part1(MLOps 개론, Model Serving, 머신러닝 프로젝트 라이프 사이클은 별도의 코드가 없으며, part2 ~ part4는 강의 자료에서 사용한 코드가 저장되어 있습니다
- `assets`엔 예제로 사용할 Mask Classification Model이 저장되어 있습니다
  - 만약 실무였다면 part 아래에 있는 폴더들에 같이 저장되는 것이 조금 더 좋지만, 교육용 Github Repo라서 중복 저장을 피하기 위해 이렇게 저장했습니다


```
├── README.md
├── .github/workflows : Github Action Workflow
├── assets : Mask Classification Model
│   └── mask_task
├── part2
│   ├── 01-voila
│   ├── 02-streamlit
│   └── 04-cicd
├── part3
│   ├── 01-fastapi
│   ├── 02-docker
│   ├── 03-logging
│   └── 04-mlflow
└── part4
    ├── 01-bentoml
    └── 02-airflow
```


<br />

## 추천 학습 방식
- 강의를 수강한 후, 강의에서 나온 공식 문서를 보며 코드를 작성합니다
- 강의에서 활용한 코드를 그대로 복사 붙여넣기하지 말고, **직접 타이핑해주세요**
  - 오류를 많이 경험하고, 오류가 발생했다면 그 이유와 해결 방법을 별도로 기록해주세요
- 강의에서 진행된 코드를 더 좋게 개선해도 좋아요 :)
- 강의에서 다룬 내용보다 더 넓은 내용을 찾아보고 정리해서 나만의 Serving 노트를 만들어보아요


<br />

## 만약 이슈가 생겼다면
- 실습 코드 관련 문의는 **슬랙 채널**에 문의해주세요
  - 어떤 상황에서(OS, Python 버전 등) 어떤 오류가 발생했는지 자세히 알려주시면 좋아요. 같이 해결해보아요!
- 강의 영상 및 강의 자료 관련 문의는 **부스트코스**를 통해서 해주세요!
- 이해하기 어렵거나 토론하고 싶은 소재가 있다면 **슬랙 채널**에 공유해주세요!
  - 단순히 GET이 뭐에요? 라는 질문보다는 간단히 검색한 후, 검색한 자료를 공유하면서 어떤 부분이 어려운지 질문주시는 것을 추천드려요 :)

<br />

## 협업 규칙

- 커밋 메시지 컨벤션은 [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/)을 따릅니다 
  - [commitizen](https://github.com/commitizen-tools/commitizen)을 사용하면 더욱 쉽게 커밋할 수 있습니다
- 작업은 기본적으로 별도의 브랜치를 생성하여 작업합니다. 작업이 완료되면 PR로 리뷰 받습니다
- PR 리뷰 후 머지 방식은 Squash & Merge를 따릅니다
  - Merge 전에 PR 제목을 되도록이면 convetional commit 형태로 만들어주세요



<br />

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="http://zzsza.github.io"><img src="https://avatars.githubusercontent.com/u/18207755?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Sung Yun Byeon</b></sub></a><br /><a href="#projectManagement-zzsza" title="Project Management">📆</a> <a href="#maintenance-zzsza" title="Maintenance">🚧</a> <a href="https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/commits?author=zzsza" title="Code">💻</a> <a href="https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/commits?author=zzsza" title="Documentation">📖</a></td>
    <td align="center"><a href="https://codethief.io"><img src="https://avatars.githubusercontent.com/u/12247655?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jungwon Seo</b></sub></a><br /><a href="https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/commits?author=thejungwon" title="Code">💻</a> <a href="#content-thejungwon" title="Content">🖋</a> <a href="https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/commits?author=thejungwon" title="Documentation">📖</a> <a href="#example-thejungwon" title="Examples">💡</a></td>
    <td align="center"><a href="https://humphreyahn.dev/"><img src="https://avatars.githubusercontent.com/u/24207964?v=4?s=100" width="100px;" alt=""/><br /><sub><b>humphrey</b></sub></a><br /><a href="https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/commits?author=ahnsv" title="Code">💻</a> <a href="#content-ahnsv" title="Content">🖋</a> <a href="https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/commits?author=ahnsv" title="Documentation">📖</a> <a href="#example-ahnsv" title="Examples">💡</a></td>
    <td align="center"><a href="http://dailyheumsi.tistory.com"><img src="https://avatars.githubusercontent.com/u/31306282?v=4?s=100" width="100px;" alt=""/><br /><sub><b>heumsi</b></sub></a><br /><a href="https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/commits?author=heumsi" title="Code">💻</a> <a href="#content-heumsi" title="Content">🖋</a> <a href="https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/commits?author=heumsi" title="Documentation">📖</a> <a href="#example-heumsi" title="Examples">💡</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/ykmoon/"><img src="https://avatars.githubusercontent.com/u/45195471?v=4?s=100" width="100px;" alt=""/><br /><sub><b>YkMoon</b></sub></a><br /><a href="https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/commits?author=Ykmoon" title="Code">💻</a> <a href="#content-Ykmoon" title="Content">🖋</a> <a href="https://github.com/zzsza/Boostcamp-AI-Tech-Product-Serving/commits?author=Ykmoon" title="Documentation">📖</a> <a href="#example-Ykmoon" title="Examples">💡</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!