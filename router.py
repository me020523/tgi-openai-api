#!/usr/bin/env python3
import time
import argparse
import json
import secrets
import time
from typing import AsyncGenerator, Generator, Optional, Union, Dict, List, Any

import tiktoken
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from generate import ModelServer
from loguru import logger
from sentence_transformers import SentenceTransformer

from constants import ErrorCode
from protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
    EmbeddingsResponse,
    EmbeddingsRequest,
    FunctionCallResponse,
)
from react_prompt import get_qwen_react_prompt

app = FastAPI()
headers = {"User-Agent": "Chat API Server"}
args: argparse.Namespace = None
embed_client: SentenceTransformer = None


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=500
    )


def check_requests(request) -> Optional[JSONResponse]:
    # Check all params
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.n is not None and request.n <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.stop is not None and (
        not isinstance(request.stop, str) and not isinstance(request.stop, list)
    ):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )

    return None


def get_gen_params(
    model_name: str,
    messages: Union[str, List[Dict[str, str]]],
    *,
    temperature: Optional[float],
    top_p: Optional[float],
    n: Optional[int],
    max_tokens: Optional[int],
    stream: Optional[bool],
    stop: Optional[Union[str, List[str]]] = None,
    functions: Optional[List[Dict[str, Any]]] = None,
    function_call: Union[str, Dict[str, str]] = "auto",
) -> Dict[str, Any]:
    if not max_tokens:
        max_tokens = 2048

    if functions is not None:
        messages = get_qwen_react_prompt(messages, functions, function_call)

    if n is not None and n > 1:
        n = 1

    gen_params = {
        "model": model_name,
        "prompt": messages,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "max_new_tokens": max_tokens,
        "stream": stream,
    }

    if stop is not None:
        if isinstance(stop, str):
            stop = [stop]
        gen_params["stop"] = stop

    logger.debug(f"==== request ====\n{gen_params}")
    return gen_params


def chat_completion_stream_generator(model_name: str, gen_params: Dict[str, Any]):
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    _id = f"chatcmpl-{secrets.token_hex(12)}"
    finish_stream_events = []

    for content in model_server.generate_stream_gate(gen_params):
        if content["error_code"] != 0:
            yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        delta_text = content["text"]

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=delta_text),
            finish_reason=content.get("finish_reason", None),
        )
        chunk = ChatCompletionStreamResponse(
            id=_id, choices=[choice_data], model=model_name
        )

        if len(delta_text) == 0:
            if content.get("finish_reason", None) is not None:
                finish_stream_events.append(chunk)
                continue
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_none=True, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


def generate_completion_stream_generator(request: CompletionRequest):
    model_name = request.model
    _id = f"cmpl-{secrets.token_hex(12)}"
    finish_stream_events = []

    for text in request.prompt:
        payload = get_gen_params(
            request.model,
            text,
            n=1,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=request.stream,
            stop=request.stop,
        )

        for content in model_server.generate_stream_gate(payload):
            if content["error_code"] != 0:
                yield f"data: {json.dumps(content, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            delta_text = content["text"]

            choice_data = CompletionResponseStreamChoice(
                index=0,
                text=delta_text,
                logprobs=content.get("logprobs", None),
                finish_reason=content.get("finish_reason", None),
            )
            chunk = CompletionStreamResponse(
                id=_id,
                object="text_completion",
                choices=[choice_data],
                model=model_name,
            )
            if len(delta_text) == 0:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.append(chunk)
                    continue

            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def show_available_models():
    model_cards = []
    model_list = [args.model_name]
    for m in model_list:
        model_cards.append(ModelCard(id=m, root=m, permission=[ModelPermission()]))
    return ModelList(data=model_cards)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    gen_params = get_gen_params(
        request.model,
        request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stream=request.stream,
        stop=request.stop,
        n=request.n,
        functions=request.functions,
        function_call=request.function_call,
    )

    if request.stream:
        generator = chat_completion_stream_generator(
            request.model,
            gen_params,
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    choices = []
    usage = UsageInfo()
    content = model_server.generate_gate(gen_params)

    if content["error_code"] != 0:
        return create_error_response(content["error_code"], content["text"])

    if request.messages[-1]["role"] == "user" and request.functions is not None:
        react_content = content["text"].strip()
        thought_index = react_content.index("Thought:")
        name_index, arguments_index = react_content.index(
            "Action:"
        ), react_content.index("Action Input:")
        function_call = FunctionCallResponse(
            name=react_content[name_index + 8 : arguments_index].strip(),
            arguments=react_content[arguments_index + 14 :],
            thought=react_content[thought_index + 9 : name_index],
        )
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", function_call=function_call),
                finish_reason="function_call",
            )
        )
    else:
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content["text"]),
                finish_reason=content.get("finish_reason", "stop"),
            )
        )

        task_usage = UsageInfo.parse_obj(content["usage"])
        for usage_key, usage_value in task_usage.dict().items():
            if usage_key != "first_tokens":
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
                usage.first_tokens = content["usage"].get("first_tokens", None)

    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret
    start_time = time.time()
    if isinstance(request.prompt, str):
        request.prompt = [request.prompt]

    if request.stream:
        generator = generate_completion_stream_generator(request)
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        text_completions = []
        for text in request.prompt:
            gen_params = get_gen_params(
                request.model,
                text,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stream=request.stream,
                n=1,
                stop=request.stop,
            )
            content = model_server.generate_gate(gen_params)
            text_completions.append(content)

        choices = []
        usage = UsageInfo()
        for i, content in enumerate(text_completions):
            if content["error_code"] != 0:
                return create_error_response(content["error_code"], content["text"])

            choices.append(
                CompletionResponseChoice(
                    index=i,
                    text=content["text"],
                    logprobs=content.get("logprobs", None),
                    finish_reason=content.get("finish_reason", "stop"),
                )
            )

            task_usage = UsageInfo.parse_obj(content["usage"])
            for usage_key, usage_value in task_usage.dict().items():
                if usage_key != "first_tokens":
                    setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
            usage.first_tokens = content["usage"].get("first_tokens", None)

        logger.info(
            f"consume time  = {(time.time() - start_time)}s, response = {str(choices)}"
        )
        return CompletionResponse(
            model=request.model, choices=choices, usage=UsageInfo.parse_obj(usage)
        )


@app.post("/v1/embeddings")
@app.post("/v1/engines/{model_name}/embeddings")
async def create_embeddings(request: EmbeddingsRequest, model_name: str = None):
    """Creates embeddings for the text"""
    if request.model is None:
        request.model = model_name

    inputs = request.input
    if isinstance(inputs, str):
        inputs = [inputs]
    elif isinstance(inputs, list):
        if isinstance(inputs[0], int):
            decoding = tiktoken.model.encoding_for_model(request.model)
            inputs = [decoding.decode(inputs)]
        elif isinstance(inputs[0], list):
            decoding = tiktoken.model.encoding_for_model(request.model)
            inputs = [decoding.decode(text) for text in inputs]

    data, token_num = [], 0
    batches = [
        inputs[i : min(i + 1024, len(inputs))] for i in range(0, len(inputs), 1024)
    ]
    for num_batch, batch in enumerate(batches):
        payload = {
            "model": request.model,
            "input": batch,
        }

        embedding = model_server.get_embeddings(payload)

        data += [
            {
                "object": "embedding",
                "embedding": emb,
                "index": num_batch * 1024 + i,
            }
            for i, emb in enumerate(embedding["embedding"])
        ]
        token_num += embedding["token_num"]

    return EmbeddingsResponse(
        data=data,
        model=request.model,
        usage=UsageInfo(
            prompt_tokens=token_num,
            total_tokens=token_num,
            completion_tokens=None,
        ),
    ).dict(exclude_none=True)


def main(args1):
    global args
    args = args1

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.info(f"args: {args}")

    global model_server
    model_server = ModelServer(
        args.tgi_server,
        args.embedding_name,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
