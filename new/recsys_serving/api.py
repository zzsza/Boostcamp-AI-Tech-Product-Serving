import os
from sqlmodel import Session
from database import PredictionResult, engine
from pydantic import BaseModel, ValidationError
from fastapi import APIRouter, Body, HTTPException, status

from model import ModelOptions

router = APIRouter()

model_path = os.environ.get("MODEL_PATH")
data_path = os.environ.get("DATA_PATH")


class PredictionResponse(BaseModel):
    user_id: int
    isbn: str
    rating: float
    model: str


# 여러 모델에 대한 옵션, 학습 옵션을 추가해 볼 수 있습니다
# DeepCoNN 은 첫 실행 시 vector_create True 설정 필요
@router.post("/scoring")
def train_model(
    input_data: dict = Body(...),
) -> PredictionResponse:
    model_name = input_data.get("model", "").lower()
    train = input_data.get("train", True)
    vector_create = input_data.get("vector_create", False)
    model_builder = ModelOptions(model_name, data_path, model_path, accelerator="cpu")

    if model_path is None or data_path is None:
        raise ValueError("MODEL_PATH or DATA_PATH is not defined.")

    try:
        # DeepCoNN 의 경우 첫 실행 시 vector_create true 설정이 필요합니다
        # 다른 모델에 대한 옵션 값 설정도 구현해 봅시다
        embeddings = model_builder.get_embedding(vector_create)
        model_builder.load_model(embeddings=embeddings)
        model = model_builder.get_model()

        trainer = model_builder.get_trainer(model=model)
        trainer.train(embeddings) if train else None
        scores = trainer.test(embeddings)

        # sample
        response = PredictionResponse(
            user_id=11676,
            isbn='0000000000',
            rating=scores[0],
            model=model_name
        )

    except RuntimeError:
        raise HTTPException(status_code=500, detail="Model is not initialized")
    except ValueError:
        raise HTTPException(status_code=400, detail="Input is not valid")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {e}")
        # 결과를 데이터베이스에 저장

    prediction_result = PredictionResult(result=response)
    with Session(engine) as session:
        session.add(prediction_result)
        session.commit()
        session.refresh(prediction_result)

    return response


@router.get("/scoring/{user_id}")
def get_result(user_id: int) -> PredictionResponse:
    # 데이터베이스에서 특정 결과 가져옴
    with Session(engine) as session:
        prediction_result = session.get(PredictionResult, user_id)
        if not prediction_result:
            raise HTTPException(
                detail="Not found", status_code=status.HTTP_404_NOT_FOUND
            )
        return PredictionResponse(
            user_id=prediction_result.user_id,
            isbn=prediction_result.isbn,
            rating=prediction_result.rating,
            model=prediction_result.model
        )


@router.get("/scoring")
def get_results() -> list[PredictionResponse]:
    # 데이터베이스에서 결과 가져옴
    with Session(engine) as session:
        prediction_results = session.query(PredictionResult).all()
        return [
            PredictionResponse(
                user_id=prediction_result.user_id,
                isbn=prediction_result.isbn,
                rating=prediction_result.rating,
                model=prediction_result.model,
            )
            for prediction_result in prediction_results
        ]
