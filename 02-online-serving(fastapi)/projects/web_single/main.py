from fastapi import FastAPI
from contextlib import asynccontextmanager
from loguru import logger
from sqlmodel import SQLModel

from config import config
from database import engine
from dependencies import load_model
from api import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 데이터베이스 테이블 생성
    logger.info("Creating database table")
    SQLModel.metadata.create_all(engine)

    # 모델 로드
    logger.info("Loading model")
    load_model(config.model_path)
    # model.py에 존재. 역할을 분리해야 할 수도 있음 => 새로운 파일을 만들고, 거기서 load_model 구현
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(router)

@app.get("/")
def root():
    return "Hello World!"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)