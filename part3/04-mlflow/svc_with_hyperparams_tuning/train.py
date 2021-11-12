from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

import mlflow

def main():
    mlflow.sklearn.autolog()

    iris = datasets.load_iris()
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)

    with mlflow.start_run() as run:
        clf.fit(iris.data, iris.target)

if __name__ == "__main__":
    main()
