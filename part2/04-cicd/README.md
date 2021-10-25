# CICD

## Prerequisite
- Github 계정
- Google Cloud Platform 계정
## 1. SSH키를 활용한 간단한 예제

### 1. 
```
#ssh 키 파일을 생성하기 위해 이동합니다.
cd ~/.ssh
#GCP를 가입했던 이메일로 사용해주세요! (username과 나중에 연관이 있습니다.)
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
#파일 이름을 다음과 같이 작성합니다.
#Enter file in which to save the key (/home/boostcamp_jungwon/.ssh/id_rsa): github-action
#아무것도 입력하지 않고 그냥 엔터를 칩니다.
#Enter passphrase (empty for no passphrase): 
#Enter same passphrase again: 
#일반적으로는 생성한 키를 authorized_keys파일에 추가를 해야하는데, 
#Google Compute Engine의 경우 주기적으로 삭제작업이 들어갑니다.
#이 부분은 뒤에서 처리하겠습니다.
cat github-action.pub >> authorized_keys
#cat github-action 명령어로 출력된 결과를 복사
```

```
cd
sudo apt-get update
sudo apt-get install python3.8-venv -y
git config --global credential.helper store
git clone <YOUR_REPOSITORY>
cd github-action-test/part2/04-cicd
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.runOnSave true
```

```
name: CI
on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: executing remote ssh commands using ssh key
        uses: appleboy/ssh-action@master
        with:
          host: 34.64.172.212
          username: boostcamp_jungwon
          key: ${{ secrets.SSH_KEY }}
          port: 22
          script: |
            cd github-action-test/part2/04-cicd
            sh deploy.sh
          
```

## 2. Docker를 활용한 간단한 예제
TBD