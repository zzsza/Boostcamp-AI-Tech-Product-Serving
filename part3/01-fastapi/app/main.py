import os
import io
import albumentations
import albumentations.pytorch
import numpy as np

from datetime import datetime
from typing import List, Optional, Union
from fastapi import FastAPI, File, UploadFile
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

from PIL import Image

from app.model import MyEfficientNet, get_model

app = FastAPI()

available_slots = int(os.getenv("AVAILABLE_SLOTS", 3))

orders = []  # TODO(humphrey): repository 객체를 생성한다


def can_order() -> bool:
    outstanding_orders = [order for order in orders if order.status == "IN_PROGRESS"]
    return len(outstanding_orders) < available_slots


def transform_image(image_bytes: bytes):
    transform = albumentations.Compose(
        [
            albumentations.Resize(height=512, width=384),
            albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            albumentations.pytorch.transforms.ToTensorV2(),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image_array = np.array(image)
    return transform(image=image_array)["image"].unsqueeze(0)


def predict_from_image_byte(image_bytes: bytes, model: MyEfficientNet) -> List[int]:
    transformed_image = transform_image(image_bytes)
    outputs = model.forward(transformed_image)
    _, y_hat = outputs.max(1)
    return y_hat.tolist()


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
    products: List[Product] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def bill(self):
        return sum([product.price for product in self.products])


class OrderUpdate(BaseModel):
    order_id: UUID
    product_id: UUID
    result: Optional[List]


def get_order_by_id(order_id: UUID) -> Order:
    return next((order for order in orders if order.id == order_id))


def update_order_by_id(order_id: UUID, next_order: Order) -> Optional[Order]:
    updated_order = None
    for index, order in enumerate(orders):
        if order.id == order_id:
            updated_order = order.copy(update=next_order.dict(exclude_unset=True))
            orders[index] = updated_order
    return updated_order


@app.get("/order", description="주문 리스트를 가져옵니다")
async def get_orders() -> List[Order]:
    return orders


@app.get("/order/{order_id}", description="주문 정보를 가져옵니다")
async def get_order(order_id: UUID) -> Union[Order, dict]:
    try:
        return get_order_by_id(order_id=order_id)
    except StopIteration:
        return {"message": "주문 정보를 찾을 수 없습니다"}


@app.post("/order", description="주문을 요청합니다")
async def make_order(
        files: List[UploadFile] = File(...), model: MyEfficientNet = Depends(get_model)
) -> Union[Order, dict]:  # TODO(humphrey): multiple file upload를 가능하게 한다
    products = []
    for file in files:
        image_bytes = await file.read()
        inference_result = predict_from_image_byte(image_bytes=image_bytes, model=model)
        product = InferenceImageProduct(result=inference_result)
        products.append(product)
    new_order = Order(products=products)
    orders.append(new_order)
    return new_order


@app.patch("/order", description="주문을 수정합니다")
async def update_order(order_id: UUID, next_order: Order):
    return update_order_by_id(order_id=order_id, next_order=next_order)


@app.get("/bill/{order_id}", description="계산을 요청합니다")
async def get_bill(order_id: UUID):
    try:
        found_order = next((order for order in orders if order.id == order_id))
        return found_order.bill
    except StopIteration:
        return {"message": "주문 정보를 찾을 수 없습니다"}
