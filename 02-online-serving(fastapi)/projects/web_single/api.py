from fastapi import APIRouter, HTTPException, status
from schemas import PredictionRequest, PredictionResponse
from dependencies import get_model
from database import PredictionResult, engine
from sqlmodel import Session, select

router = APIRouter()

@router.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    # 모델 load
    model = get_model()    
    
    # 예측 : 여러분들의 코드 상황에 따라 구현
    prediction = int(model.predict([request.features])[0])
     
    # 예측한 결과를 DB에 저장
    # 데이터베이스 객체를 생성. 그 때 prediction을 사용
    prediction_result = PredictionResult(result=prediction)
    with Session(engine) as session:
        session.add(prediction_result)
        session.commit()
        session.refresh(prediction_result)

    return PredictionResponse(id=prediction_result.id, result=prediction)

    # return PredictionResponse

@router.get("/predict")
def get_predictions() -> list[PredictionResponse]:
    with Session(engine) as session:
        statement = select(PredictionResult)
        prediction_results = session.exec(statement).all()
        # prediction_results = session.query(PredictionResult).all()
        return [
            PredictionResponse(id=prediction_result.id, result=prediction_result.result)
            for prediction_result in prediction_results
        ]

@router.get("/predict/{id}")
def get_preidction(id: int) -> PredictionResponse:
    with Session(engine) as session:
        prediction_result = session.get(PredictionResult, id)
        if not prediction_result:
            raise HTTPException(
                detail="Not found", status_code=status.HTTP_404_NOT_FOUND
            )
        return PredictionResponse(
            id=prediction_result.id, result=prediction_result.result
        )

