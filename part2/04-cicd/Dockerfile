FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
ENTRYPOINT ["streamlit","run", "--server.port","8080"]
CMD ["app.py"]