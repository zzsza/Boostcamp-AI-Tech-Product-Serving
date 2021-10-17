import os
import io
import albumentations
import albumentations.pytorch
import numpy as np


from datetime import datetime
from typing import Any, List
from fastapi import FastAPI, File, UploadFile
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

from PIL import Image

from app.model import MyEfficientNet, get_model

app = FastAPI()

available_slots = int(os.getenv("AVAILABLE_SLOTS", 3))

orders = []

def transform_image(image_bytes):
    transform = albumentations.Compose([
            albumentations.Resize(height=512, width=384),
            albumentations.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.2, 0.2, 0.2)),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image_array = np.array(image)
    return transform(image=image_array)['image'].unsqueeze(0)

def predict_from_image_byte(image_bytes, model):
    transformed_image = transform_image(image_bytes)
    outputs = model.forward(transformed_image)
    _, y_hat = outputs.max(1)
    return transformed_image, y_hat



class Product(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    price: float

class InferenceImageProduct(Product):
    name: str = "inference_image_product"
    price: float = 100.0
    result: Any # FIXME(humphrey): tensor type일 때 json serialize가 안된다


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
async def order(file: UploadFile = File(...), model: MyEfficientNet = Depends(get_model)): 
    outstanding_orders = [order for order in orders if order.status == "IN_PROGRESS"]
    if len(outstanding_orders) >= available_slots:
        return {"message": "손님이 많습니다. 주문을 진행할 수 없습니다"}
    
    image_bytes = await file.read()
    inference_result = predict_from_image_byte(image_bytes=image_bytes, model=model)

    product = InferenceImageProduct(result=inference_result) # TODO(humphrey): output schema를 지정한다
    new_order = Order(status="STARTED", products=[product]) # TODO(humphrey): queue에 inference task를 넣고, background에 돌려야하나?

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

