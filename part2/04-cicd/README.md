# CICD

## Prerequisite
- Google Cloud Platform에서 Compute Engine을 실행해놓은 상태

## 1. SSH키를 활용한 배포

### 1. Compute Engine에 접속하여 SSH Key 생성

  1. ssh 키 파일을 생성하기 위해 이동합니다.
        ```
        cd ~/.ssh
        ```
  1. GCP를 가입했던 이메일로 사용해주세요! (username과 나중에 연관이 있습니다.)
        ```
        `ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
        ```
  1. 파일 이름을 다음과 같이 작성합니다.
      - `Enter file in which to save the key (/home/boostcamp_jungwon/.ssh/id_rsa): github-action`
  1. 아무것도 입력하지 않고 그냥 엔터를 칩니다.
      - `Enter passphrase (empty for no passphrase): `
      - `Enter same passphrase again: `
  1. 일반적으로는 생성한 키를 authorized_keys파일에 추가를 해야하는데, Google Compute Engine의 경우 주기적으로 삭제작업이 들어갑니다. 
        ```
        cat github-action.pub >> authorized_keys
        ```
        - GCP 콘솔 페이지의 메타데이터 섹션에서 SSH키를 추가합니다.
           - https://cloud.google.com/compute/docs/instances/adding-removing-ssh-keys
    

  1. `cat github-action` 명령어로 출력된 결과를 복사해두고 메모장같은 곳에 잘 보관해둡니다.
     1. 추후에 Github Secret에 저장할 내용입니다.


### 2. 서버 초기 세팅
  1. 계정의 초기경로로 이도합니다.
      ```
      cd
      ```
  3. 다음 명령어로 최소한의 패키지를 설치합니다.
      ```
      sudo apt-get update
      sudo apt-get install python3.8-venv -y
      ```
  1. 서빙할 코드를 가져옵니다. 이때 private repository인 경우에는 아래 명령어를 활용하여 인증 정보를 저장합니다.
      ```
      git config --global credential.helper store
      git clone <YOUR_REPOSITORY>
      ```
  1. 실행하려는 경로로 이동한 뒤에 virtualenv를 설정한 뒤에 웹사이트가 돌아가는지 확인합니다.
      ```
      cd <YOUR_REPOSITORY>/part2/04-cicd
      python3 -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt
      streamlit run app.py --server.runOnSave true
      ```
  1. 이때 접근이 되지 않는다면, GCP의 방화벽 설정에서 `8501`포트를 열어둡니다.

### 3. Github Action 세팅
1. 배포하려는 Repo에서 Setting으로 이동 한 뒤에 SECRET섹션에서 github action에서 사용할 값들을 저장합니다 
   - `GCP_SSH_KEY`라는 값을 추가합니다.
     - 이때 내용에는 아까 복사한 `github-action` 키를 붙여 넣습니다.
     - `BEGIN`으로 시작하는 부분부터 `END`까지모두 복사 붙여넣기 합니다.
   - `GCP_HOST`라는 값을 추가합니다.
     - 이때 서버의 public ip주소를 넣습니다.
   - `GCP_USERNAME`이라는 값을 추가합니다.
     - 이때 위에서 키를 만들때 사용했던 유저 이름을 추가합니다.

2. 만약 아직 Github Action을 세팅하지 않았다면, Github Repo페이지에서 Github Action탭을 클릭한 다음에, `set up a workflow yourself`를 클릭 한뒤에 아래 내용을 붙여넣기 합니다. 
   
    ```
    name: CICD-SSH
    on:
    push:
        paths:
            - 'part2/04-cicd/**'

    jobs:
    build:
        runs-on: ubuntu-latest
        steps:
        - name: executing remote ssh commands using ssh key
        uses: appleboy/ssh-action@master
        with:
            host: ${{ secrets.GCP_HOST }} 
            username: ${{ secrets.GCP_USERNAME }}
            key: ${{ secrets.GCP_SSH_KEY }}
            port: 22
            script: |
                cd github-action-test/part2/04-cicd
                sh deploy.sh
            
    ```

### 4. 테스트
Github에서 `app.py` 파일의 title 부분을 변경하여서, 현재 돌고 있는 서버에 업데이트가 반영되는지 확인합니다.
```
st.title("Mask Classification Model CICD TEST")
```
## 2. Docker를 활용한 간단한 예제

### 1. GCP 서비스를 이용하기 위한 서비스 계정생성
   1. GCP 콘솔에서 IAM 및 관리자 페이지로 이동
   1. 서비스 계정으로 이동
   1. 서비스 계정 생성
   1. 서비스 계정 이름만 생성하고, 자동으로 생성되는 ID를 사용하셔도 됩니다.
   1. 생성한 계정을 클릭하고, 키 탭으로 이동 한뒤 새로운 키를 json 타입으로 생성한 뒤 다운로드
### 2. 서비스 계정에 역할 부여
   1. IAM 페이지로 이동한 뒤, 위에서 만든 계정을 수정
   1. 서비스 계정 `사용자`, 저장소 `관리자`, Cloud Run `관리자` 역할 추가
### 3. Github Secrets 등록
   - 첫번째 과정에서 얻은, json 키파일의 내용을 복사 후에 `SERVICE_ACCOUNT_KEY` 이름으로 등록
   - GCP 콘솔에서 본인의 project id(프로젝트 이름 x)를 복사 후에 `GCP_PROJECT_ID` 이름으로 등록
### 4. Github Action 세팅
- `.github/workflows` 폴더에 아래 내용으로 yml file 생성
    ```
    name: Build and Push Python Image to Google Cloud Platform
    on:
    push:
        paths:
            - 'part2/04-cicd/**'
    jobs:
    build-push-gcr:
        name: Build and Push to GCP
        runs-on: ubuntu-latest
        env:
        APP_NAME: streamlit
        IMAGE_NAME: gcr.io/${{ secrets.GCP_PROJECT_ID }}/streamlit
        steps:
        - name: Checkout repository
            uses: actions/checkout@v2
        - name: Login
            uses: google-github-actions/setup-gcloud@master
            with:
            project_id: ${{ secrets.GCP_PROJECT_ID }}
            service_account_key: ${{ secrets.SERVICE_ACCOUNT_KEY }}
        - name: Configure Docker
            run: gcloud auth configure-docker --quiet

        - name: Move model file
            run: |
            cd part2/04-cicd
            sh deploy_docker.sh

        - name: Build Docker image
            run: docker build part2/04-cicd -t $IMAGE_NAME

        - name: Push Docker image
            run: docker push $IMAGE_NAME

        - name: Deploy Docker image
            run: gcloud run deploy ${{ secrets.GCP_PROJECT_ID }} --image $IMAGE_NAME --region asia-northeast3 --platform managed --memory 4096Mi
    ```
### 5. 테스트
- Github에서 `app.py` 파일의 title 부분을 변경하여서, 현재 돌고 있는 서버에 업데이트가 반영되는지 확인합니다.
    ```
    st.title("Mask Classification Model DOCKER TEST")
    ```
- Github action 로그를 보면 `Service URL: https://***-xxxxxxxx-du.a.run.app`와 같은 형태로 나오는데, ***부분에 여러분들의 project id를 입력해주시면 접근 가능합니다.