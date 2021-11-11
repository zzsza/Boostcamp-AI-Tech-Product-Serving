from fastapi import FastAPI, HTTPException

app = FastAPI()

items = {
    1: "Boostcamp",
    2: "AI",
    3: "Tech"
}


@app.get("/v1/{item_id}")
async def find_by_id(item_id: int):
    return items[item_id]


@app.get("/v2/{item_id}")
async def find_by_id(item_id: int):
    try:
        item = items[item_id]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"아이템을 찾을 수 없습니다 [id: {item_id}]")
    return item
