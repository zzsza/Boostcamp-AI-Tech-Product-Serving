# 1. simple long-running tasks
import contextlib
import json
import threading
import time
from datetime import datetime
from time import sleep
from typing import List

import requests
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()


def run_tasks_in_fastapi(app: FastAPI, tasks: List):
    """
    FastAPI Client를 실행하고, task를 요청합니다

    Returns:
        List: responses
    """
    config = uvicorn.Config(app, host="127.0.0.1", port=5000, log_level="error")
    server = Server(config=config)
    with server.run_in_thread():
        responses = []
        for task in tasks:
            response = requests.post("http://127.0.0.1:5000/task", data=json.dumps(task))
            if not response.ok:
                continue
            responses.append(response.json())
    return responses


app_1 = FastAPI()


def cpu_bound_task(wait_time: int):
    sleep(wait_time)
    return f"task done after {wait_time}"


class TaskInput(BaseModel):
    wait_time: int = Field(default=1, le=10, ge=1)


@app_1.post("/task")
def create_task(task_input: TaskInput):
    return cpu_bound_task(task_input.wait_time)


tasks = [{"wait_time": i} for i in range(1, 10)]

start_time = datetime.now()
run_tasks_in_fastapi(app_1, tasks)
end_time = datetime.now()
print(f"Simple Tasks: Took {(end_time - start_time).seconds}")

# 2. background tasks
app_2 = FastAPI()


@app_2.post("/task",
            status_code=202)  # 비동기 작업이 등록됐을 때, HTTP Response 202 (Accepted)를 보통 리턴합니다. https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/202
async def create_task_in_background(task_input: TaskInput, background_tasks: BackgroundTasks):
    background_tasks.add_task(cpu_bound_task, task_input.wait_time)
    return "ok"


start_time = datetime.now()
run_tasks_in_fastapi(app_2, tasks)
end_time = datetime.now()
print(f"Background Tasks: Took {(end_time - start_time).seconds}")

# 3. background tasks with in-memory task repo
from uuid import UUID, uuid4

app_3 = FastAPI()


class TaskInput2(BaseModel):
    id_: UUID = Field(default_factory=uuid4)
    wait_time: int


task_repo = {}


def cpu_bound_task_2(id_: UUID, wait_time: int):
    sleep(wait_time)
    result = f"task done after {wait_time}"
    task_repo[id_] = result


@app_3.post("/task", status_code=202)
async def create_task_in_background_2(task_input: TaskInput2, background_tasks: BackgroundTasks):
    background_tasks.add_task(cpu_bound_task_2, id_=task_input.id_, wait_time=task_input.wait_time)
    return task_input.id_


@app_3.get("/task/{task_id}")
def get_task_result(task_id: UUID):
    try:
        return task_repo[task_id]
    except KeyError:
        return None
