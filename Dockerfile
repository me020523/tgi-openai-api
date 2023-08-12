FROM me020523/text-generation-inference:1.0.1
    MAINTAINER me020523 <me020523@gmail.com>

ENV TGI_SERVER=http://127.0.0.1:8080
ENV EMBEDDIG_MODEL=BAAI/bge-large-en
ENV SENTENCE_TRANSFORMERS_HOME=/st_cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "app.py", "--host", "0.0.0.0", \
    "--port", "80", \
    "--tgi_server", "${TGI_SERVER}", \
    "--embedding_name", "${EMBEDDING_MODEL}"]
