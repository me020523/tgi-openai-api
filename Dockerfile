FROM python:3.10-alpine
    MAINTAINER me020523 <me020523@gmail.com>

ENV TGI_SERVER="http://127.0.0.1:8080"
ENV EMBEDDIG_MODEL="BAAI/bge-large-en"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py", "--host", "0.0.0.0", \
    "--port", "8080", \
    "--tgi_server", "${TGI_SERVER}", \
    "--embedding_name", "${EMBEDDING_MODEL}"]