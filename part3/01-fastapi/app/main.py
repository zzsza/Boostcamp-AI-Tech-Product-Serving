import os
import random
from datetime import datetime
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

app = FastAPI()

available_slots = int(os.getenv("AVAILABLE_SLOTS", 3))

orders = []


class Product(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    price: float

class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    status: str
    products: List[Product] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def bill(self):
        return sum([product.price for product in self.products])



@app.get("/order", description="주문 리스트를 가져옵니다")
async def get_orders():
    return orders


@app.get("/order/{order_id}", description="주문 정보를 가져옵니다")
async def get_order(order_id: UUID):
    try:
        return next((order for order in orders if order.id == order_id))
    except StopIteration:
        return {"message": "주문 정보를 찾을 수 없습니다"}


@app.post("/order", description="주문을 요청합니다", status_code=202)
async def order(): # TODO(humphrey): 메뉴를 정의한다
    outstanding_orders = [order for order in orders if order.status == "IN_PROGRESS"]
    if len(outstanding_orders) >= available_slots:
        return {"message": "손님이 많습니다. 주문을 진행할 수 없습니다"}

    new_order = Order(status="STARTED", )
    orders.append(new_order)
    return new_order.id


@app.patch("/order", description="주문을 수정합니다")
async def update_order():
    pass # TODO(humphrey): 주문 상태를 수정한다


@app.get("/bill/{order_id}", description="계산을 요청합니다")
async def get_bill(order_id: UUID):
    try:
        found_order = next((order for order in orders if order.id == order_id))
        return found_order.bill
    except StopIteration:
        return {"message": "주문 정보를 찾을 수 없습니다"}

