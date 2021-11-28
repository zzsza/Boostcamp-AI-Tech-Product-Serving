from fastapi import FastAPI
import uvicorn

app = FastAPI()

items = {}


@app.on_event("startup")
def startup_event():
    print("Start Up Event")
    items["foo"] = {"name": "Fighters"}
    items["bar"] = {"name": "Tenders"}


@app.on_event("shutdown")
def shutdown_event():
    print("Shutdown Event!")
    with open("log.txt", mode="a") as log:
        log.write("Application shutdown")


@app.get("/items/{item_id}")
def read_items(item_id: str):
    return items[item_id]


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)


