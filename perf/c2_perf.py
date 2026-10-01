#!/usr/bin/env python3
"""Performance test of the two C2 backends: the decision server (systemone) and a chat model.

Runs inside the cluster (a pod in the model namespace), with the Python standard library only.
It sends the same requests as the router (questions and system prompt from
litellm/privacy_scoring.py, copied next to this file) and prints one JSON line per measurement
plus a summary table.

Scenarios:
  concurrency  C2 calls with 1..N parallel clients, short unique prompts
  size         C2 latency by prompt size (about 400 to 40,000 tokens)
  busy         C2 latency while the chat model generates long answers (the agents' load)

Nothing is changed in the cluster: the script only sends requests.

  python3 c2_perf.py \
      --decision-url http://dgemma-decision-predictor.local-models.svc.cluster.local/v1 \
      --chat-url http://qwen38-local-predictor.local-models.svc.cluster.local/v1 \
      --chat-model qwen38-local
"""

import argparse
import json
import random
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from privacy_scoring import PrivacyScorer as S

WORDS = (
    "the service team review deployment latency budget customer contract migration database "
    "cluster network storage policy release incident report schedule meeting analysis vendor "
    "invoice project roadmap quarter target security audit training feedback process manager "
    "colleague planning support request ticket upgrade backup monitor alert dashboard metric"
).split()
SENSITIVE = (" My colleague told me in confidence that she is going through a divorce and may "
             "lose her job next month.")


def text_of(tokens, sensitive=False):
    """A unique text of about `tokens` tokens (about 1.3 tokens per word); unique, so the
    prefix cache of vLLM cannot help."""
    rnd = random.Random(time.time_ns())
    words = [str(rnd.randrange(10**6))] + [rnd.choice(WORDS) for _ in range(int(tokens / 1.3))]
    return " ".join(words) + (SENSITIVE if sensitive else "")


def post(url, body, timeout):
    t0 = time.perf_counter()
    headers = {"content-type": "application/json"}
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return data, time.perf_counter() - t0


class Backends:
    def __init__(self, args):
        self.args = args
        self.questions = {
            q: {"type": "noul", "instructions": t}
            for q, t in {**S._SYSTEMONE_POSITIVE, **S._SYSTEMONE_IGNORED}.items()
        }

    # systemone-auto: no "samples" key, the server default ("auto": 1 to 4 noise draws, more when
    # the answer is uncertain). systemone-s1: "samples": 1.
    def systemone(self, text, samples=None):
        body = {"model": "dgemma", "state": {"text": text}, "questions": self.questions}
        if samples:
            body["samples"] = samples
        url = self.args.decision_url.rstrip("/") + "/systemone"
        data, secs = post(url, body, self.args.timeout)
        a = data["answers"]
        p = max(a["personal_sensitive"]["noul"], a["credentials"]["noul"])
        return secs, data.get("usage", {}).get("input_tokens"), p

    def chat(self, text):
        body = {
            "model": self.args.chat_model, "max_tokens": 200, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": S._CLASSIFIER_SYSTEM},
                         {"role": "user", "content": f"<text>\n{text}\n</text>"}],
        }
        url = self.args.chat_url.rstrip("/") + "/chat/completions"
        data, secs = post(url, body, self.args.timeout)
        content = data["choices"][0]["message"]["content"] or ""
        return secs, data.get("usage", {}).get("prompt_tokens"), '"sensitive": true' in content

    def call(self, backend, text):
        if backend == "systemone-auto":
            return self.systemone(text)
        if backend == "systemone-s1":
            return self.systemone(text, samples=1)
        return self.chat(text)


def summarize(lat):
    lat = sorted(lat)
    if not lat:
        return {"n": 0}
    return {"n": len(lat), "p50": round(statistics.median(lat) * 1000),
            "p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))] * 1000),
            "max": round(lat[-1] * 1000)}


def emit(rec):
    print(json.dumps(rec), flush=True)


def run_batch(b, backend, n, conc, tokens, sensitive=False):
    lat, errors, toks = [], [], []

    def one(_):
        try:
            secs, t, _p = b.call(backend, text_of(tokens, sensitive))
            lat.append(secs)
            toks.append(t)
        except Exception as exc:
            errors.append(type(exc).__name__)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        list(ex.map(one, range(n)))
    wall = time.perf_counter() - t0
    toks = [t for t in toks if t]
    return {**summarize(lat), "errors": len(errors), "error_types": sorted(set(errors)),
            "rps": round(len(lat) / wall, 2),
            "tokens": round(statistics.median(toks)) if toks else None}


def scenario_concurrency(b, backends, levels):
    for backend in backends:
        for conc in levels:
            res = run_batch(b, backend, n=max(20, conc * 4), conc=conc, tokens=400)
            emit({"scenario": "concurrency", "backend": backend, "concurrency": conc, **res})


def scenario_size(b, backends, sizes):
    for backend in backends:
        for size in sizes:
            res = run_batch(b, backend, n=5, conc=1, tokens=size, sensitive=True)
            emit({"scenario": "size", "backend": backend, "size": size, **res})


def scenario_busy(b, backends, streams_levels, args):
    """Background: `streams` threads ask the chat model for long answers (thinking on) on
    prompts of `--busy-prompt` tokens, as the agents do. Foreground: sequential C2 calls on
    each backend."""
    for streams in streams_levels:
        stop = threading.Event()
        gen_tokens = []

        def generate(stop=stop, gen_tokens=gen_tokens):
            prompt = "Write a detailed incident analysis for: " + text_of(args.busy_prompt)
            body = {"model": args.chat_model, "max_tokens": 1500, "stream": False,
                    "messages": [{"role": "user", "content": prompt}]}
            while not stop.is_set():
                try:
                    data, _ = post(args.chat_url.rstrip("/") + "/chat/completions", body, 600)
                    gen_tokens.append(data.get("usage", {}).get("completion_tokens", 0))
                except Exception:
                    time.sleep(1)

        threads = [threading.Thread(target=generate, daemon=True) for _ in range(streams)]
        for t in threads:
            t.start()
        time.sleep(args.busy_warmup)
        for backend in backends:
            res = run_batch(b, backend, n=args.busy_calls, conc=1, tokens=400)
            emit({"scenario": "busy", "backend": backend, "chat_streams": streams,
                  "busy_prompt": args.busy_prompt, **res})
        stop.set()
        emit({"scenario": "busy-load", "chat_streams": streams, "answers_done": len(gen_tokens)})
        for t in threads:
            t.join(timeout=650)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-url", required=True)
    ap.add_argument("--chat-url", required=True)
    ap.add_argument("--chat-model", required=True)
    ap.add_argument("--scenarios", default="concurrency,size,busy")
    ap.add_argument("--backends", default="systemone-auto,systemone-s1,chat")
    ap.add_argument("--levels", default="1,2,4,8,16,32")
    ap.add_argument("--sizes", default="400,4000,16000,40000")
    ap.add_argument("--streams", default="4,8")
    ap.add_argument("--busy-calls", type=int, default=15)
    ap.add_argument("--busy-warmup", type=int, default=30)
    ap.add_argument("--busy-prompt", type=int, default=600, help="tokens of each background prompt")
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()
    b = Backends(args)
    backends = args.backends.split(",")
    for backend in backends:  # warm-up: the first call after a start is slow
        b.call(backend, text_of(50))
    scen = args.scenarios.split(",")
    if "concurrency" in scen:
        scenario_concurrency(b, backends, [int(x) for x in args.levels.split(",")])
    if "size" in scen:
        scenario_size(b, backends, [int(x) for x in args.sizes.split(",")])
    if "busy" in scen:
        scenario_busy(b, backends, [int(x) for x in args.streams.split(",")], args)


if __name__ == "__main__":
    main()
