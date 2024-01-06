from fastapi import APIRouter, HTTPException, status, UploadFile
from pydantic import BaseModel
from sqlmodel import Session
from PIL import Image
from torchvision.transforms import transforms

from database import PredictionResult, engine
from dependencies import get_model

router = APIRouter()


class PredictionResponse(BaseModel):
    id: int
    result: int


# FastAPI 경로
@router.post("/predict")
def predict(file: UploadFile) -> PredictionResponse:
    # 이미지 파일이 맞는지 검증
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            detail="File type not supported", status_code=status.HTTP_400_BAD_REQUEST
        )
    image = Image.open(file.file)

    # 이미지 파일을 텐서로 변환
    image_tensor = transforms.ToTensor()(image)

    # 텐서의 크기가 맞는지 검증
    if image_tensor.shape != (3, 512, 384):
        raise HTTPException(
            detail="Image size must be 384x512", status_code=status.HTTP_400_BAD_REQUEST
        )

    # 모델 추론
    model = get_model()
    prediction = int(model(image_tensor.unsqueeze(0)).argmax())

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
