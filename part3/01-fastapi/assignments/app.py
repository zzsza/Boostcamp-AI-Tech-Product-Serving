from typing import Optional, List

from fastapi import FastAPI, HTTPException
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


class UpdateModelIn(BaseModel):
    version: str
    description: Optional[str]
    tags: List[str]
    artifact_url: str


@app.get("/models")
def get_models():
    # TODO: model 리스트를 리턴합니다
    return models


@app.get("/model/{model_id}")
def get_model(model_id: int):
    # TODO: model 리스트로 부터 model_id가 일치하는 model을 가져와 리턴합니다
    for model in models:
        if model.id == model_id:
            return model
    # TODO: 일치하는 model_id가 없을 때 404 에러와 에러 메시지를 추가해봅니다.
    raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다")


@app.get("/model")
def get_model_by_name(model_name: str):
    # TODO: model 리스트로 부터 model_name이 일치하는 model을 가져와 리턴합니다
    for model in models:
        if model.name == model_name:
            return model
    raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다")


@app.post("/model")  # TODO: CreateModelOut이라는 class를 만들고, 새로운 모델의 id 만을 응답하도록 바꿔보기
def create_model(new_model: CreateModelIn):
    # TODO: model을 새로 만들고 model 리스트에 저장합니다
    model = Model(**new_model.dict(), id=len(models) + 1)
    models.append(model)
    return model.id


@app.patch("/model/{model_id}")  # TODO: UpdateModelOut이라는 class를 만들어서, 업데이트된 모델의 id를 제외한 모든 정보 보여주도록 바꿔보기
def update_model(model_id: int, update_data: UpdateModelIn):
    # TODO: 매칭되는 model_id를 가지고 있는 모델을 업데이트합니다

    try:
        existing_model_idx, existing_model = next(
            ((model_idx, model) for model_idx, model in enumerate(models) if model.id == model_id), None)
    except TypeError:
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다")

    updated_model = existing_model.copy()
    for key, value in update_data.dict(exclude_unset=True).items():
        setattr(updated_model, key, value)
    models[existing_model_idx] = updated_model
    return updated_model


@app.delete("/model/{model_id}", status_code=204)  # TODO: status code를 204로 바꿔보기
def delete_model(model_id: int):
    # TODO: 매칭되는 model_id를 가지고 있는 모델을 model 리스트로 부터 삭제합니다
    for model_idx, model in enumerate(models):
        if model.id == model_id:
            del models[model_idx]


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0")
