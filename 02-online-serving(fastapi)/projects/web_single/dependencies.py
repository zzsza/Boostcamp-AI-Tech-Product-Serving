model = None

def load_model(model_path: str):
    import joblib
    
    global model
    model = joblib.load(model_path)

def get_model():
    global model
    return model