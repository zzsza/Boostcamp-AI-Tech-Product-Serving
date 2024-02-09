import datetime
from sqlmodel import SQLModel, Field, create_engine
from config import config


class PredictionResult(SQLModel,table=True):
    id: int = Field(default=None, primary_key=True)
    result: int
    created_at: str = Field(default_factory=datetime.datetime.now)
    # default_factory : default를 설정. 동적으로 값을 지정. 

engine = create_engine(config.db_url)