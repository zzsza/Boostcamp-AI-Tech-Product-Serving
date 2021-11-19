# bento_packer.py
# 모델 학습
from sklearn import svm
from sklearn import datasets

clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

# bento_service.py에서 정의한 IrisClassifier
from bento_service import IrisClassifier

# IrisClassifier 인스턴스 생성
iris_classifier_service = IrisClassifier()

# Model Artifact를 Pack
iris_classifier_service.pack('model', clf)

# Model Serving을 위한 서비스를 Disk에 저장
saved_path = iris_classifier_service.save()
