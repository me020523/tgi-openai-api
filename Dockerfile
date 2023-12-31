FROM me020523/text-generation-inference:1.0.1
    MAINTAINER me020523 <me020523@gmail.com>

ENV TGI_SERVER=http://127.0.0.1:8080
ENV EMBEDDING_MODEL=BAAI/bge-large-en
ENV EMBEDDING_DEVICE=cpu
ENV SENTENCE_TRANSFORMERS_HOME=/st_cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
