#!/usr/bin/env python3
import argparse
import json
import sys

sys.path.insert(0, ".")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OpenAI Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--tgi_server", default="127.0.0.1:8080", help="tgi server address"
    )
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser.add_argument(
        "--embedding_name",
        help="embedding model name or path",
        type=str,
        default="BAAI/bge-large-en",
    )
    parser.add_argument(
        "--embedding_device", help="embedding device", type=str, default="cpu"
    )
    args = parser.parse_args()

    from router import main

    main(args)
