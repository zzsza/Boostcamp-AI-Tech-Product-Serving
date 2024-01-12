from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, \
    AutoModelForSequenceClassification

router = APIRouter()

model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-small")
tokenizer = AutoTokenizer.from_pretrained('klue/roberta-small', max_length=160)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: str
    score: float
    text: str


@router.post("/classify/")
async def classify_text(classify_in: PredictionRequest) -> PredictionResponse:
    try:
        result = classifier(classify_in.text)
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Model is not initialized")
    except ValueError:
        raise HTTPException(status_code=400, detail="Input is not valid")
    except Exception:
        raise HTTPException(status_code=500, detail="Something went wrong")

    # Extract the predicted label and confidence score
    label = result[0]['label']
    score = result[0]['score']

    return PredictionResponse(
        label=label,
        score=score,
        text=classify_in.text
    )
