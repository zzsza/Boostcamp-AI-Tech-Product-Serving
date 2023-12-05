from typing import Any, Dict

from fastapi import FastAPI, Depends, File

from model import MyEfficientNet, get_model, predict_from_image_byte, get_config

app = FastAPI()


@app.post("/predict")
async def predict(image_file: bytes = File(...), model: MyEfficientNet = Depends(get_model),
                  config: Dict[str, Any] = Depends(get_config)):
    return predict_from_image_byte(model=model, image_bytes=image_file, config=config)
