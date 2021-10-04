import os
import random
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

available_slots = int(os.getenv("AVAILABLE_SLOTS", 3))

orders = []


class Order(BaseModel):
    id: int
    status: str
    created_at: datetime = Field(default_factory=datetime.now)


@app.get("/order", description="주문 리스트를 가져옵니다")
async def get_orders():
    return orders


@app.get("/order/{order_id}", description="주문 정보를 가져옵니다")
async def get_order(order_id: int):
    return next((order for order in orders if order.id == order_id))


@app.post("/order", description="주문을 요청합니다", status_code=202)
async def order():
    outstanding_orders = [order for order in orders if order.status == "IN_PROGRESS"]
    if len(outstanding_orders) >= available_slots:
        return {"message": "손님이 많습니다. 주문을 진행할 수 없습니다"}

    new_order = Order(id=random.randint(1, 10000), status="STARTED", )
    orders.append(new_order)
    return new_order.id


@app.patch("/order", description="주문을 수정합니다")
async def update_order():
    pass


@app.get("/bill/{order_id}", description="계산을 요청합니다")
async def get_bill(order_id: int):
    pass
