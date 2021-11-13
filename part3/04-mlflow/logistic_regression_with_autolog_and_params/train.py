# train.py

import argparse
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    mlflow.sklearn.autolog()

    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])

    lr = LogisticRegression(solver=sys.argv[1], penalty=sys.argv[2], l1_ratio=float(sys.argv[3]))

    with mlflow.start_run() as run:
        lr.fit(X, y)

    score = lr.score(X, y)
    print("Score: %s" % score)
   
