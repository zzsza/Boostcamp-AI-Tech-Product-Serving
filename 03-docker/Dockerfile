FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

COPY . /app
WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN pip install pip==23.0.1 && \
    pip install poetry==1.2.1 && \
    poetry export -o requirements.txt && \
    pip install -r requirements.txt 

CMD ["python", "main.py"]
