from sqlmodel import Field, SQLModel, create_engine

from config import config


class PredictionResult(SQLModel, table=True):
    user_id: int = Field(default=999, primary_key=True)
    isbn: str
    rating: float


engine = create_engine(config.db_url)
