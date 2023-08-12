FROM ghcr.io/huggingface/text-generation-inference:1.0.0
    MAINTAINER me020523 <me020523@gmail.com>

ENV TGI_SERVER="http://127.0.0.1:8080"
ENV EMBEDDIG_MODEL="BAAI/bge-large-en"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "app.py", "--host", "0.0.0.0", \
    "--port", "8000", \
    "--tgi_server", "${TGI_SERVER}", \
    "--embedding_name", "${EMBEDDING_MODEL}"]
