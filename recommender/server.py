import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from .engine import Recommender


def create_server(engine, host="127.0.0.1", port=8000):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/health":
                return self.respond(200, {"status": "ok", "items": len(engine.items)})
            if url.path != "/recommendations":
                return self.respond(404, {"error": "not found"})
            params = parse_qs(url.query)
            user_id = params.get("user_id", [""])[0]
            if not user_id.strip():
                return self.respond(400, {"error": "user_id is required"})
            try:
                limit = int(params.get("limit", ["10"])[0])
                results = engine.recommend(user_id, limit)
            except ValueError:
                return self.respond(400, {"error": "limit must be an integer between 1 and 100"})
            self.respond(200, {"user_id": user_id, "recommendations": results})

        def respond(self, status, body):
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description="Recommendation HTTP API")
    parser.add_argument("--data", default="data/interactions.csv")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = create_server(Recommender(args.data), args.host, args.port)
    print(f"Listening on {args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
