from typing import Optional, List

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

app = FastAPI()


class Model(BaseModel):
    id: int
    name: str
    version: str
    description: Optional[str]
    tags: List[str]
    artifact_url: str


models: List[Model] = []


class CreateModelIn(BaseModel):
    name: str
    version: str
    description: Optional[str]
    tags: List[str]
    artifact_url: str


class CreateModelOut(BaseModel):
    id: int


class UpdateModelIn(BaseModel):
    version: str
    description: Optional[str]
    tags: List[str]
    artifact_url: str


class UpdateModelOut(BaseModel):
    name: str
    version: str
    description: Optional[str]
    tags: List[str]
    artifact_url: str


@app.get("/models")
def get_models():
    return models


@app.get("/model/{model_id}")
def get_model(model_id: int):
    for model in models:
        if model.id == model_id:
            return model
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"모델을 찾을 수 없습니다  model_id: {model_id}")
    # 위의 코드를 좀더 간결하게+파이써닉하게 만들려면,
    # model = next((model for model in models if model.id == model_id), None)
    # if not model:
    #   raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"모델을 찾을 수 없습니다\tmodel_id: {model_id}")
    # return model


@app.get("/model")
def get_model_by_name(model_name: str):
    for model in models:
        if model.name == model_name:
            return model
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"모델을 찾을 수 없습니다  model_name: {model_name}")


@app.post("/model", response_model=CreateModelOut)
def create_model(new_model: CreateModelIn):
    model = Model(id=len(models) + 1, **new_model.dict())
    models.append(model)
    return model


@app.patch("/model/{model_id}",
           response_model=UpdateModelOut)
def update_model(model_id: int, update_data: UpdateModelIn):
    model_index = -1
    existing_model: Optional[Model] = None
    for idx, model in enumerate(models): # NOTE: 반복되는 로직이 보이지 않으신가요? 이런 부분은 다른 함수로 만들면 더 간결하게 코드를 만들 수 있습니다.
        if model.id == model_id:
            existing_model = model
            model_index = idx

    if not existing_model and model_index == -1:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"모델을 찾을 수 없습니다  model_id: {model_id}")

    updated_model = existing_model.copy()
    for key, value in update_data.dict().items():
        setattr(updated_model, key, value)
    models[model_index] = updated_model
    return updated_model


@app.delete("/model/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(model_id: int):
    id_to_delete = -1
    for idx, model in enumerate(models):
        if model.id == model_id:
            id_to_delete = idx

    if id_to_delete == -1:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"모델을 찾을 수 없습니다  model_id: {model_id}")

    del models[id_to_delete]
    return ""

