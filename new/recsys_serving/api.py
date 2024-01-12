import os
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from model import ModelOptions

router = APIRouter()

model_path = os.environ.get("MODEL_PATH")
data_path = os.environ.get("DATA_PATH")


class PredictionResponse(BaseModel):
    user_id: int
    isbn: str
    rating: float


# 여러 모델에 대한 옵션, 학습 옵션을 추가해 볼 수 있습니다
# DeepCoNN 은 첫 실행 시 vector_create True 설정 필요
# 요청 예시: 0.0.0.0:8000/scoring/context?model_type=wdn
@router.get("/scoring/context")
def predict(model_type: str = "FM") -> PredictionResponse:
    model_name = model_type.lower()
    model_options = ModelOptions(model_name, data_path, model_path, accelerator="cpu")

    if model_path is None or data_path is None:
        raise ValueError("MODEL_PATH or DATA_PATH is not defined.")

    try:
        embeddings = model_options.get_embedding()
        model_options.load_model(embeddings=embeddings)
        model = model_options.get_model()

        trainer = model_options.get_trainer(
            model=model,
        )
        trainer.train(embeddings)
        scores = trainer.test(embeddings)

        # sample
        response = PredictionResponse(
            user_id=11676,
            isbn='0000000000',
            rating=scores[0]
        )

    except RuntimeError:
        raise HTTPException(status_code=500, detail="Model is not initialized")
    except ValueError:
        raise HTTPException(status_code=400, detail="Input is not valid")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {e}")

    return response
