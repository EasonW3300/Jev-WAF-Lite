"""Local v2-only HTTP UI and prediction API."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

# The strict v2 loader rejects legacy bundles, while predict_loaded shares preprocessing with training.
from security_jev_v2.model import load_bundle
from security_jev_v2.predict import predict_loaded
from security_jev_v2.train import device_name, synchronize

PAGE = Path(__file__).with_name("app.html")
MAX_BYTES = 64 * 1024


class SecurityEngine:
    def __init__(self, bundle: Path, device: str = "auto"):
        self.device = device_name(device)
        self.model, self.tokenizer, self.manifest, self.policy = load_bundle(bundle, self.device)
        self.lock = threading.Lock()

    def info(self) -> dict:
        return {"model_version": self.manifest["model_version"], "device": self.device,
                "model_source": self.manifest["model_source"],
                "encoder_initialization": self.manifest.get("encoder_initialization"),
                "selected_epoch": self.manifest.get("selected_epoch"),
                "selected_step": self.manifest.get("selected_step"),
                "training_data_sha256": self.manifest.get("data_sha256", {}).get("train"),
                "max_length": self.manifest["max_length"], "api_version": 2,
                "production_validated": False}

    def predict(self, state: str) -> dict:
        if not isinstance(state, str) or not state.strip():
            raise ValueError("请输入 HTTP 请求文本")
        if len(state.encode("utf-8")) > MAX_BYTES:
            raise ValueError("请求文本最多 64 KiB")
        with self.lock:
            started = time.perf_counter()
            result = predict_loaded(self.model, self.tokenizer, self.manifest,
                                    self.policy, state, self.device)
            synchronize(self.device)
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return result


def make_server(host: str, port: int, engine: SecurityEngine) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format_string, *args):
            return

        def send_body(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, result: dict):
            self.send_body(status, json.dumps(result, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self.send_body(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/v2/info":
                return self.send_json(200, engine.info())
            return self.send_json(404, {"error": "页面不存在"})

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/api/v2/predict":
                return self.send_json(404, {"error": "接口不存在"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > MAX_BYTES * 6 + 1024:
                    raise ValueError("JSON 请求体过大")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("请求体必须是 JSON 对象")
                self.send_json(200, engine.predict(body.get("state")))
            except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                print(f"[v2] {type(exc).__name__}: {exc}", file=sys.stderr)
                self.send_json(500, {"error": "推理失败，请查看本地终端"})

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/bert-base-v2"))
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    engine = SecurityEngine(args.bundle, args.device)
    server = make_server(args.host, args.port, engine)
    print(f"v2 本地页面：http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
