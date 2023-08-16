#!/usr/bin/env python3
from torch.cuda import OutOfMemoryError
from constants import ErrorCode
import torch
from sentence_transformers import SentenceTransformer
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
)

from text_generation.client import Response, StreamResponse
from text_generation.types import FinishReason
from protocol import Role
from text_generation import Client


server_error_msg = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)


class PromptConverter:
    def convert(self, messages: List[Dict[str, str]]) -> str:
        return ""


class AlpacaPromptConverter(PromptConverter):
    def convert(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == Role.USER:
                prompt += f"\n\n### Input:\n{content}"
            if role == Role.SYSTEM:
                prompt += f"\n\n### Instruction:\n{content}"
            if role == Role.ASSISTANT:
                prompt += f"\n\n### Response:\n{content}"
        prompt += "\n\n### Response:\n"
        return prompt


class AlpacaInstructOnlyPromptConverter(PromptConverter):
    def convert(self, messages: List[Dict[str, str]]) -> str:
        prompt: str = ""
        system: Optional[str] = None
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == Role.USER:
                if system is None:
                    prompt += f"### Instruction:\n\n{content}\n\n"
                else:
                    prompt += f"### Instruction:\n\n{system}\n{content}\n\n"
                    system = None
            if role == Role.SYSTEM:
                system = content
            if role == Role.ASSISTANT:
                prompt += f"### Response:\n{content}\n"
        prompt += "### Response:\n"
        return prompt


class StableBelugaPromptConverter(PromptConverter):
    def convert(self, messages: List[Dict[str, str]]) -> str:
        promot = ""

        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == Role.USER:
                promot += f"\n\n### User:\n{content}"
            elif role == Role.SYSTEM:
                promot += f"\n\n### System:\n{content}"
            elif role == Role.ASSISTANT:
                promot += f"\n\n### Assistant:\n{content}"

        promot += "\n\n### Assistant:\n"

        return promot


class VicunaPromptConventer(PromptConverter):
    def convert(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""

        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == Role.USER:
                prompt += f"USER: {content}\n"
            elif role == Role.SYSTEM:
                prompt += f"{content}\n\n"
            elif role == Role.ASSISTANT:
                prompt += f"ASSISTANT:{content}\n"

        prompt += "ASSISTANT:"
        return prompt


class ModelServer:
    def __init__(
        self,
        tgi_server: str,
        model_name: str = "alpaca",
        embedding_name: str = "BAAI/bge-large-en",
        embedding_device: str = "cpu",
    ):
        self.tgi_server = tgi_server
        self.model_name = model_name
        self.embedding_name = embedding_name
        self.embed_device = embedding_device
        self.tgi_client = Client(self.tgi_server)
        self.embed_client = SentenceTransformer(embedding_name, device=embedding_device)
        self.prompt_converter = {
            "alpaca": AlpacaPromptConverter(),
            "stable-beluga": StableBelugaPromptConverter(),
            "vicuna": VicunaPromptConventer(),
            "platypus2": AlpacaPromptConverter(),
        }

    def setup_completion_prompt(self, prompt: str) -> List[Dict[str, str]]:
        return [
            {
                "role": Role.SYSTEM,
                "content": "You will be provided with texts, and your task is to complete the text to make it meaningful.",
            },
            {
                "role": Role.USER,
                "content": prompt,
            },
        ]

    def generate_prompt(self, messages: List[Dict[str, str]]) -> str:
        if self.model_name not in self.prompt_converter:
            c = AlpacaPromptConverter()
        else:
            c = self.prompt_converter[self.model_name]

        return c.convert(messages)

    def to_openai_finish_reason(self, r: FinishReason) -> str:
        if r == FinishReason.Length:
            return "length"
        elif r == FinishReason.StopSequence:
            return "stop"
        else:
            return "stop"

    def generate_gate(self, params) -> Dict:
        if not isinstance(params["prompt"], list):
            messages = self.setup_completion_prompt(params["prompt"])
        else:
            messages = params["prompt"]

        resp: Response = self.tgi_client.generate(
            self.generate_prompt(messages),
            do_sample=True,
            max_new_tokens=params["max_new_tokens"],
            best_of=params["n"],
            temperature=params["temperature"],
            top_p=None
            if params["top_p"] >= 1 or params["top_p"] <= 0
            else params["top_p"],
        )

        return {
            "error_code": 0,
            "text": resp.generated_text,
            "finish_reason": self.to_openai_finish_reason(resp.details.finish_reason),
            "usage": {
                "completion_tokens": resp.details.generated_tokens,
                "prompt_tokens": len(resp.details.prefill),
                "total_tokens": len(resp.details.prefill)
                + resp.details.generated_tokens,
            },
        }

    def generate_stream_gate(self, params):
        if not isinstance(params["prompt"], list):
            messages = self.setup_completion_prompt(params["prompt"])
        else:
            messages = params["prompt"]

        resp_iter: Iterator[StreamResponse] = self.tgi_client.generate_stream(
            self.generate_prompt(messages),
            do_sample=True,
            max_new_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"]
            if params["top_p"] > 0 and params["top_p"] < 1
            else None,
        )

        for resp in resp_iter:
            ret: Dict[str, Any] = {"error_code": 0}

            if not resp.token.special:
                ret["text"] = resp.token.text
            else:
                ret["text"] = ""

            if resp.details is not None:
                ret["finish_reason"] = self.to_openai_finish_reason(
                    resp.details.finish_reason
                )
                ret["usage"] = {
                    "completion_tokens": resp.details.generated_tokens,
                }
            yield ret

    @torch.inference_mode()
    def get_embeddings(self, params):
        try:
            embeddings = self.embed_client.encode(
                params["input"], normalize_embeddings=True
            )
            ret = {
                "embedding": embeddings.tolist(),
                "token_num": sum([len(i) for i in params["input"]]),
            }
        except OutOfMemoryError as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret
