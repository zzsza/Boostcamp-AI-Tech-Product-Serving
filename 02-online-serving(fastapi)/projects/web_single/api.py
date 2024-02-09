from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from sqlmodel import Session

from database import PredictionResult, engine
from dependencies import get_model

router = APIRouter()


class PredictionRequest(BaseModel):
    features: list


class PredictionResponse(BaseModel):
    id: int
    result: int


# FastAPI 경로
@router.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    # 모델 추론
    model = get_model()
    prediction = int(model.predict([request.features])[0])

    # 결과를 데이터베이스에 저장
    prediction_result = PredictionResult(result=prediction)
    with Session(engine) as session:
        session.add(prediction_result)
        session.commit()
        session.refresh(prediction_result)

    # 응답
    return PredictionResponse(id=prediction_result.id, result=prediction)


@router.get("/predict/{id}")
def get_prediction(id: int) -> PredictionResponse:
    # 데이터베이스에서 결과를 가져옴
    with Session(engine) as session:
        prediction_result = session.get(PredictionResult, id)
        if not prediction_result:
            raise HTTPException(
                detail="Not found", status_code=status.HTTP_404_NOT_FOUND
            )
        return PredictionResponse(
            id=prediction_result.id, result=prediction_result.result
        )


@router.get("/predict")
def get_predictions() -> list[PredictionResponse]:
    # 데이터베이스에서 결과를 가져옴
    with Session(engine) as session:
        prediction_results = session.query(PredictionResult).all()
        return [
            PredictionResponse(id=prediction_result.id, result=prediction_result.result)
            for prediction_result in prediction_results
        ]
