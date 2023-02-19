# QnA

과거에 나왔던 질문, 답변 사례에 대해 정리해둡니다.

## poetry는 conda와 상관없이 사용하는건가요?

poetry는 의존성을 관리해주는 도구에요. conda나 pyenv 등이 있지만 버전의 의존성을 잘 관리해주진 않고 깨지는 경우가 존재합니다.

이를 위해 요새 파이썬 백엔드에선 poetry를 많이 사용하고 있어요!

conda 같은 경우 처음 공부할 때는 좋으나, 현업에서 쓰기엔 conda는 너무 무겁습니다. 그래서 miniconda를 쓰기도 하는데, 저는 conda로 환경 구성하는 것을 비선호하곤 해요. 연구만 한다고 하면 conda도 괜찮을 것 같은데 프러덕션 레벨까지 고려하시면 conda보단 pytion 버전 관리하는 방법(pyenv, virtualenv, poetry의 조합)을 추천드리고 싶어요.

강의에서 poetry로 넣은 이유는 협업을 미리 경험해보길 원하는 마음에 넣었다고 봐주세요.

## GCP를 만드는 계정과 github 계정이 동일해야하는 걸까요? 

아니요. 달라도 상관없습니다.

## git clone 과정에서 permission denied 창이 뜹니다.

Private Repo 인가요?

Private Repo 클론 받는 방법은 다음 두가지가 있는데요.

- https 클론 받는 경우 -> github access token 사용
- ssh 클론 받는 경우 -> pub, private 키 사용

보통 후자로 많이 합니다.