#!/usr/bin/env sh

python app.py \
    --host "0.0.0.0" \
    --port "80" \
    --tgi_server "${TGI_SERVER}" \
    --embedding_name "${EMBEDDING_MODEL}" \
    --embedding_device "${EMBEDDING_DEVICE}"
