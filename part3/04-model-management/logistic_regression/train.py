import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])

    penalty = "elasticnet"
    l1_ratio = 0.1
    lr = LogisticRegression(penalty=penalty, l1_ratio=l1_ratio, solver="saga")

    lr.fit(X, y)

    score = lr.score(X, y)
    print("Score: %s" % score)
   
    mlflow.log_param("penalty", penalty)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model")
