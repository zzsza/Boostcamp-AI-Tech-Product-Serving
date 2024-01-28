from fastapi import FastAPI
import uvicorn

app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


@app.get("/items/")
def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip: skip + limit]


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)



