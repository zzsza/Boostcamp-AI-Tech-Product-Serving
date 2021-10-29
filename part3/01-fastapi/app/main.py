from datetime import datetime
from typing import List, Optional, Union
from fastapi import FastAPI, File, UploadFile
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

from app.model import MyEfficientNet, get_model, predict_from_image_byte

app = FastAPI()

orders = []  # TODO(humphrey): repository 객체를 생성한다


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
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def bill(self):
        return sum([product.price for product in self.products])


class OrderUpdate(BaseModel):
    products: List[Product] = Field(default_factory=list)


def get_order_by_id(order_id: UUID) -> Order:
    return next((order for order in orders if order.id == order_id))


def update_order_by_id(order_id: UUID, next_products: List[Product]) -> Optional[Order]:
    updated_order = None
    for index, order in enumerate(orders):
        if order.id == order_id:
            updated_order: Order = order.copy()
            existing_product_ids = [product.id for product in updated_order.products]
            new_products = [next_product for next_product in next_products if
                            next_product.id not in existing_product_ids]
            if not new_products:
                break
            updated_order.products += new_products
            updated_order.updated_at = datetime.now()
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


@app.patch("/order/{order_id}", description="주문을 수정합니다")
async def update_order(order_id: UUID, order_update: OrderUpdate):
    return update_order_by_id(order_id=order_id, next_products=order_update.products)


@app.get("/bill/{order_id}", description="계산을 요청합니다")
async def get_bill(order_id: UUID):
    try:
        found_order = next((order for order in orders if order.id == order_id))
        return found_order.bill
    except StopIteration:
        return {"message": "주문 정보를 찾을 수 없습니다"}
