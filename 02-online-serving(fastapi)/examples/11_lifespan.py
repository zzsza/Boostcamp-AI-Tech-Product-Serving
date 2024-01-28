from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn

items = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Start Up Event")
    items["foo"] = {"name": "Fighters"}
    items["bar"] = {"name": "Tenders"}

    yield

    print("Shutdown Event!")
    with open("log.txt", mode="a") as log:
        log.write("Application shutdown")


app = FastAPI(lifespan=lifespan)


@app.get("/items/{item_id}")
def read_items(item_id: str):
    return items[item_id]


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)