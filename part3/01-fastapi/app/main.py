import os

from datetime import datetime
from typing import List, Optional, Union
from fastapi import FastAPI, File, UploadFile
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

from app.model import MyEfficientNet, get_model, predict_from_image_byte

app = FastAPI()

available_slots = int(os.getenv("AVAILABLE_SLOTS", 3))

orders = []  # TODO(humphrey): repository 객체를 생성한다


def can_order() -> bool:
    outstanding_orders = [order for order in orders if order.status == "IN_PROGRESS"]
    return len(outstanding_orders) < available_slots


class Product(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    price: float


class InferenceImageProduct(Product):
    name: str = "inference_image_product"
    price: float = 100.0
    result: Optional[List]


class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    status: str
    products: List[Product] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def bill(self):
        return sum([product.price for product in self.products])


@app.get("/order", description="주문 리스트를 가져옵니다")
async def get_orders() -> List[Order]:
    return orders


@app.get("/order/{order_id}", description="주문 정보를 가져옵니다")
async def get_order(order_id: UUID) -> Union[Order, dict]:
    try:
        return next((order for order in orders if order.id == order_id))
    except StopIteration:
        return {"message": "주문 정보를 찾을 수 없습니다"}


@app.post("/order", description="주문을 요청합니다", status_code=202)
async def order(
    file: UploadFile = File(...), model: MyEfficientNet = Depends(get_model)
) -> Union[Order, dict]:  # TODO(humphrey): multiple file upload를 가능하게 한다
    if not can_order():
        return {"message": "손님이 많습니다. 주문을 진행할 수 없습니다"}

    product = InferenceImageProduct(result=None)
    new_order = Order(status="IN_PROGRESS", products=[product])
    orders.append(new_order)

    image_bytes = await file.read()
    # TODO(humphrey): background에서 prediction과 domain model update를 수행한다
    inference_result = predict_from_image_byte(image_bytes=image_bytes, model=model)
    # TODO(humphrey): update order 함수를 활용해서 코드를 간소화한다
    existing_order = await get_order(new_order.id)
    if isinstance(existing_order, dict):
        return existing_order
    existing_order.status = "DONE"
    existing_order.products[0].result = inference_result

    return existing_order


@app.patch("/order", description="주문을 수정합니다")
async def update_order():
    pass  # TODO(humphrey): 주문 상태를 수정한다


@app.get("/bill/{order_id}", description="계산을 요청합니다")
async def get_bill(order_id: UUID):
    try:
        found_order = next((order for order in orders if order.id == order_id))
        return found_order.bill
    except StopIteration:
        return {"message": "주문 정보를 찾을 수 없습니다"}
