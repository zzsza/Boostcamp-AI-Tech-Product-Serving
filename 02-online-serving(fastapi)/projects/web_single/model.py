from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def get_dataset():
    iris = load_iris()
    X, y = iris.data, iris.target
    return X, y


def get_model():
    model = RandomForestClassifier(n_estimators=100)
    return model


def train(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    return model.predict(X_test)


def evaluate(model, X_test, y_test):
    return model.score(X_test, y_test)


def save_model(model, model_path: str):
    import joblib

    joblib.dump(model, model_path)


def load_model(model_path: str):
    import joblib

    return joblib.load(model_path)


def main():
    X, y = get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = get_model()
    model = train(model, X_train, y_train)
    score = evaluate(model, X_test, y_test)
    print(f"model score: {score}")
    save_model(model, "model.joblib")


if __name__ == "__main__":
    main()
