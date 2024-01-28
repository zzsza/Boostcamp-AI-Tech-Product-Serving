from fastapi import FastAPI, APIRouter
import uvicorn

user_router = APIRouter(prefix="/users")
order_router = APIRouter(prefix="/orders")


@user_router.get("/", tags=["users"])
def read_users():
    return [{"username": "Rick"}, {"username": "Morty"}]


@user_router.get("/me", tags=["users"])
def read_user_me():
    return {"username": "fakecurrentuser"}


@user_router.get("/{username}", tags=["users"])
def read_user(username: str):
    return {"username": username}


@order_router.get("/", tags=["orders"])
def read_orders():
    return [{"order": "Taco"}, {"order": "Burritto"}]


@order_router.get("/me", tags=["orders"])
def read_order_me():
    return {"my_order": "taco"}


@order_router.get("/{order_id}", tags=["orders"])
def read_order_id(order_id: str):
    return {"order_id": order_id}


app = FastAPI()

if __name__ == '__main__':
    app.include_router(user_router)
    app.include_router(order_router)
    uvicorn.run(app, host="0.0.0.0", port=8000)


