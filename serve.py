"""HTTP front end. Stdlib only; one request at a time.

    python serve.py --port 8077

POST /v1/decide  {"state": ..., "questions": {name: {type, instructions, criteria}}}
                 -> {"answers": {name: Answer}, "timing": {...}}; 400 on bad input
GET  /healthz    -> {"ok": true, "format": ..., "load_seconds": ...}
"""
from __future__ import annotations

import argparse, json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer

from cribl_decision import FORMATS, Question
from cribl_mlx import CriblDecisionModelMLX


def handler(model: CriblDecisionModelMLX):
    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/healthz":
                return self._send(404, {"error": "not found"})
            self._send(200, {"ok": True, "format": model.fmt.name,
                             "load_seconds": round(model.load_seconds, 1)})

        def do_POST(self):
            if self.path != "/v1/decide":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                qs = {n: Question(q["type"], q["instructions"], q.get("criteria"))
                      for n, q in req["questions"].items()}
                # ValueError also covers format limits (e.g. >26 letter candidates)
                answers, timing = model.ask(req["state"], qs)
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                return self._send(400, {"error": f"{type(e).__name__}: {e}"})
            self._send(200, {"answers": {n: asdict(a) for n, a in answers.items()},
                             "timing": timing})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--format", default="plain_letter_qfirst", choices=list(FORMATS))
    a = ap.parse_args()
    model = CriblDecisionModelMLX(fmt=a.format)
    # ponytail: single-threaded on purpose; one model, one GPU, requests queue.
    srv = HTTPServer((a.host, a.port), handler(model))
    print(f"cribl-decision on http://{a.host}:{a.port} (format {a.format}, "
          f"loaded in {model.load_seconds:.1f}s)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
