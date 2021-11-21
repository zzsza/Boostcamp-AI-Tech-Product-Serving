import asyncio
from datetime import datetime
from enum import IntEnum
from typing import List, Optional, Union, Any, Dict
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

from app.model import MyEfficientNet, get_model, predict_from_image_byte, get_config

app = FastAPI()

orders = []  # TODO(humphrey): repository 객체를 생성한다


class Product(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    price: float
    output: Any
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def update_output(self, output) -> ...:
        ...


class InferenceImageProduct(Product):
    name: str = "inference_image_product"
    price: float = 100.0
    input_image: UploadFile
    output: Optional[List]

    def update_output(self, output) -> None:
        self.output = output
        self.updated_at = datetime.now()

    class Config:
        arbitrary_types_allowed = True


class OrderStatus(IntEnum):
    PENDING = 0
    IN_PROGRESS = 1
    DONE = 2


class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    products: List[Product] = Field(default_factory=list)
    status: OrderStatus
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def bill(self) -> int:
        return sum([product.price for product in self.products])

    def add_product(self, product: Product) -> "Order":
        if product.id in [existing_product.id for existing_product in self.products]:
            return self

        self.products.append(product)
        self.updated_at = datetime.now()
        return self

    def update_status(self, status: OrderStatus) -> None:
        self.status = status
        self.updated_at = datetime.now()


class OrderUpdate(BaseModel):
    products: List[Product] = Field(default_factory=list)


def get_order_by_id(order_id: UUID) -> Optional[Order]:
    """
    Order ID로 Order를 조회합니다

    Args:
        order_id (UUID): order id

    Returns:
        Optional[Order]:

    """
    return next((order for order in orders if order.id == order_id), None)


def update_order_by_id(order_id: UUID, order_update: OrderUpdate) -> Optional[Order]:
    """
    Order를 업데이트 합니다

    Args:
        order_id (UUID): order id
        order_update (OrderUpdate): Order Update DTO

    Returns:
        Optional[Order]: 업데이트 된 Order 또는 None
    """
    existing_order = get_order_by_id(order_id=order_id)
    if not existing_order:
        return

    updated_order = existing_order.copy()
    for next_product in order_update.products:
        updated_order = existing_order.add_product(next_product)
    return updated_order


async def get_prediction_result(order_id: UUID, model, config):
    order = get_order_by_id(order_id=order_id)
    order.update_status(status=OrderStatus.IN_PROGRESS)
    await asyncio.sleep(3)
    for product in order.products:
        if not getattr(product, "input_image"):
            continue
        input_image: UploadFile = product.input_image
        image_bytes = await input_image.read()
        inference_result = predict_from_image_byte(
            image_bytes=image_bytes, model=model, config=config
        )
        product.update_output(output=inference_result)
    order.update_status(status=OrderStatus.DONE)
    return update_order_by_id(
        order_id=order_id, order_update=OrderUpdate(products=order.products)
    ).id


@app.get("/order", description="주문 리스트를 가져옵니다")
async def get_orders() -> List[Order]:
    return orders


@app.get("/order/{order_id}", description="주문 정보를 가져옵니다")
async def get_order(order_id: UUID) -> Union[Order, dict]:
    order = get_order_by_id(order_id=order_id)
    if not order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return order


@app.post("/order", description="주문을 요청합니다")
async def make_order(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model: MyEfficientNet = Depends(get_model),
    config: Dict[str, Any] = Depends(get_config),
) -> Union[UUID, dict]:
    new_order = Order(
        products=[InferenceImageProduct(input_image=file) for file in files],
        status=OrderStatus.PENDING,
    )
    background_tasks.add_task(
        get_prediction_result, order_id=new_order.id, model=model, config=config
    )
    orders.append(new_order)
    return new_order.id


@app.patch("/order/{order_id}", description="주문을 수정합니다")
async def update_order(order_id: UUID, order_update: OrderUpdate) -> Union[Order, dict]:
    updated_order = update_order_by_id(order_id=order_id, order_update=order_update)
    if not updated_order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return updated_order


@app.get("/bill/{order_id}", description="계산을 요청합니다")
async def get_bill(order_id: UUID) -> Union[float, dict]:
    found_order = get_order_by_id(order_id=order_id)
    if not found_order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return found_order.bill
