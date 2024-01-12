from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
from sqlmodel import SQLModel

from api import router
from database import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create Database
    logger.info("Creating database tables")
    SQLModel.metadata.create_all(engine)

    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
