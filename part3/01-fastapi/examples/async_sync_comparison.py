import asyncio
import logging
import time

from fastapi import FastAPI

app = FastAPI()

logger = logging.getLogger(__file__)


@app.get("/sync/sync")
def sync_operation():  # takes 10 sec
    time.sleep(10)
    logger.info('running...')


@app.get("/async/async")
async def async_operation():  # takes 10 sec
    await asyncio.sleep(10)
    logger.info('running...')


@app.get("/async/sync")
async def async_with_sync():  # takes 30 sec
    time.sleep(10)
    logger.info('running...')


@app.get("/async/sync_pool")
async def async_with_sync_threadpool():  # takes 10 sec
    from fastapi.concurrency import run_in_threadpool
    await run_in_threadpool(time.sleep, 10)
    logger.info('running...')
