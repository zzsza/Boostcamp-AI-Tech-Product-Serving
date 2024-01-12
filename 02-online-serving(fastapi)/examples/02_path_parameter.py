from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/users/{user_id}")
def get_user(user_id):
    return {"user_id": user_id}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)



