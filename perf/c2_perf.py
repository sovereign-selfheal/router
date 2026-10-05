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
  parallel-context  1..N agents at the same time: parallel Presidio and C2 calls with one
               agent context each (Presidio needs --presidio-url, as for context)
  context      agents' large contexts: Presidio /analyze and C2 latency by context size, with
               log-like text (needs --presidio-url, so it runs in a LiteLLM pod: only LiteLLM may
               reach Presidio)
  gate         the whole privacy gate, PrivacyScorer.score() with the policy --policy, as the
               router calls it: 1..N agents at the same time, each with a new agent context. The
               environment selects the detectors, as in the router (NER_ENABLED for C1, the
               CLASSIFIER_* variables for C2). Needs PyYAML and Presidio: run it in a LiteLLM pod

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


AGENT_LINES = (
    "{ts} pod/checkout-api-{n} in namespace shop-{n2} restarted: OOMKilled, exit code 137",
    "{ts} Warning BackOff kubelet Back-off restarting failed container payment in pod payment-{n}",
    "{ts} ERROR c.e.checkout.PaymentClient - HTTP 503 from https://payments.internal:8443/v2/charge"
    " (attempt {n2}/5, trace {hexid})",
    "{ts} INFO  Deployment checkout-api rolled out revision {n2} image"
    " quay.io/shop/checkout:{n}.{n2}",
    "Ticket INC-{n}: customer orders failing at checkout since {ts}; priority P{p};"
    " assigned to SRE on call",
    "{ts} node ip-10-0-{n2}-{n}.us-east-2.compute.internal condition MemoryPressure=True",
    "Tool result: kubectl get events -n shop-{n2} returned {n} events; the last 5 are listed above",
)


def agent_text(tokens, needle=None, where="middle"):
    """A context like the agents send: logs, events and ticket text from their tools (about 25
    tokens per line). `needle`: a sensitive sentence placed at the start, middle or end."""
    rnd = random.Random(time.time_ns())
    lines = []
    for _ in range(max(1, int(tokens / 25))):
        lines.append(rnd.choice(AGENT_LINES).format(
            ts=f"2026-10-01T{rnd.randrange(24):02d}:{rnd.randrange(60):02d}:{rnd.randrange(60):02d}Z",
            n=rnd.randrange(10, 9999), n2=rnd.randrange(1, 99), p=rnd.randrange(1, 4),
            hexid=f"{rnd.getrandbits(64):016x}"))
    if needle:
        pos = {"start": 0, "middle": len(lines) // 2, "end": len(lines)}[where]
        lines.insert(pos, needle)
    return "\n".join(lines)


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
        # --questions: a JSON file {id: {"instructions": ..., "ignored": bool}}, the
        # classifier.systemone.questions of a policy; default: the built-in questions.
        if getattr(args, "questions", ""):
            with open(args.questions) as fh:
                spec = json.load(fh)
        else:
            spec = {**{q: {"instructions": t} for q, t in S._SYSTEMONE_POSITIVE.items()},
                    **{q: {"instructions": t, "ignored": True}
                       for q, t in S._SYSTEMONE_IGNORED.items()}}
        self.questions = {q: {"type": "noul", "instructions": v["instructions"]}
                          for q, v in spec.items()}
        self.positive = [q for q, v in spec.items() if not v.get("ignored")]

    # systemone-auto: no "samples" key, the server default ("auto": 1 to 4 noise draws, more when
    # the answer is uncertain). systemone-s1: "samples": 1.
    def systemone(self, text, samples=None):
        body = {"model": "dgemma", "state": {"text": text}, "questions": self.questions}
        if samples:
            body["samples"] = samples
        url = self.args.decision_url.rstrip("/") + "/systemone"
        data, secs = post(url, body, self.args.timeout)
        a = data["answers"]
        p = max(a[q]["noul"] for q in self.positive)
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


QUESTIONS_LABEL = "built-in"


def emit(rec):
    print(json.dumps({**rec, "questions": QUESTIONS_LABEL}), flush=True)


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


def scenario_context(b, args):
    """By context size: Presidio /analyze (the request of the router, one language) and the two
    C2 backends. One call at a time, 3 calls per size; errors and timeouts are recorded."""
    entities = [e for e in args.presidio_entities.split(",") if e]
    presidio_on = True
    for size in [int(x) for x in args.context_sizes.split(",")]:
        if not presidio_on:
            emit({"scenario": "context", "backend": "presidio", "size": size,
                  "skipped": f"a smaller size took more than {args.presidio_stop_after} s"})
        lat, errors, found, chars = [], [], [], 0
        for _ in range(3 if presidio_on else 0):
            text = agent_text(size, SENSITIVE.strip())
            chars = len(text)
            body = {"text": text, "language": "en"}
            if entities:
                body["entities"] = entities
            try:
                data, secs = post(args.presidio_url, body, args.presidio_timeout)
                lat.append(secs)
                found.append(len(data))
            except Exception as exc:
                errors.append(type(exc).__name__)
        if presidio_on:
            emit({"scenario": "context", "backend": "presidio", "size": size, "chars": chars,
                  **summarize(lat), "errors": len(errors), "error_types": sorted(set(errors)),
                  "entities_found": round(statistics.median(found)) if found else None})
            # Presidio runs one worker: very large texts slow down its /health. Stop growing.
            if errors or (lat and max(lat) > args.presidio_stop_after):
                presidio_on = False
        for backend in args.backends.split(","):
            lat, errors, toks, probs = [], [], [], []
            for _ in range(3):
                try:
                    secs, t, p = b.call(backend, agent_text(size, SENSITIVE.strip()))
                    lat.append(secs)
                    toks.append(t)
                    probs.append(p)
                except Exception as exc:
                    errors.append(type(exc).__name__)
            toks = [t for t in toks if t]
            emit({"scenario": "context", "backend": backend, "size": size, **summarize(lat),
                  "errors": len(errors), "error_types": sorted(set(errors)),
                  "tokens": round(statistics.median(toks)) if toks else None,
                  "sensitive": [p if isinstance(p, bool) else round(p, 2) for p in probs]})


def scenario_parallel_context(b, args):
    """Agents at the same time: `conc` parallel calls with one agent context of
    `--parallel-size` tokens each, to Presidio (one worker: the calls queue) and to each C2
    backend. Two rounds per level."""
    entities = [e for e in args.presidio_entities.split(",") if e]
    size = args.parallel_size
    targets = (["presidio"] if args.presidio_url else []) + args.backends.split(",")
    for target in targets:
        for conc in [int(x) for x in args.parallel_levels.split(",")]:
            lat, errors = [], []

            def one(_, target=target, lat=lat, errors=errors):
                text = agent_text(size, SENSITIVE.strip())
                try:
                    if target == "presidio":
                        body = {"text": text, "language": "en"}
                        if entities:
                            body["entities"] = entities
                        _data, secs = post(args.presidio_url, body, args.presidio_timeout)
                    else:
                        secs, _t, _p = b.call(target, text)
                    lat.append(secs)
                except Exception as exc:
                    errors.append(type(exc).__name__)

            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=conc) as ex:
                list(ex.map(one, range(conc * 2)))
            wall = time.perf_counter() - t0
            emit({"scenario": "parallel-context", "backend": target, "size": size,
                  "concurrency": conc, **summarize(lat), "errors": len(errors),
                  "error_types": sorted(set(errors)), "rps": round(len(lat) / wall, 2)})


def scenario_gate(args):
    """The whole privacy gate: `conc` agents at the same time, each with a new agent context of
    `--gate-size` tokens, half of them with a sensitive sentence. One PrivacyScorer per thread,
    built from the policy file, so the gate runs as in the router: rules, Presidio when C1 is
    on, C2 in the gray zone. `--gate-rounds` rounds per level."""
    import asyncio

    import yaml  # only this scenario: it runs in a LiteLLM pod, which has PyYAML

    with open(args.policy) as fh:
        policy = yaml.safe_load(fh)
    threshold = args.gate_threshold
    local = threading.local()

    def scorer():
        if not hasattr(local, "scorer"):
            local.scorer = S(policy)
        return local.scorer

    ner = "on" if scorer()._ner.get("enabled") else "off"
    for conc in [int(x) for x in args.gate_levels.split(",")]:
        lat, chars, errors = [], [], []
        outcomes = {"local": 0, "sota": 0, "ner_error": 0, "c2_error": 0, "c2_fallback": 0}

        def one(i, lat=lat, chars=chars, errors=errors, outcomes=outcomes):
            text = agent_text(args.gate_size, SENSITIVE.strip() if i % 2 else None)
            t0 = time.perf_counter()
            try:
                score, signals = asyncio.run(scorer().score(text, threshold))
            except Exception as exc:
                errors.append(type(exc).__name__)
                return
            lat.append(time.perf_counter() - t0)
            chars.append(len(text))
            labels = [(src, label) for src, label, _w in signals]
            outcomes["local" if score >= threshold else "sota"] += 1
            outcomes["ner_error"] += any(s == "ner" and lb.startswith("error") for s, lb in labels)
            outcomes["c2_error"] += any(s == "classifier" and lb == "error" for s, lb in labels)
            outcomes["c2_fallback"] += any(lb.startswith("fallback/") for _s, lb in labels)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as ex:
            list(ex.map(one, range(conc * args.gate_rounds)))
        wall = time.perf_counter() - t0
        emit({"scenario": "gate", "ner": ner, "size": args.gate_size, "threshold": threshold,
              "chars": round(statistics.median(chars)) if chars else None,
              "concurrency": conc, **summarize(lat), **outcomes, "errors": len(errors),
              "error_types": sorted(set(errors)), "rps": round(len(lat) / wall, 2)})


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
    ap.add_argument("--presidio-url", default="")
    ap.add_argument("--presidio-timeout", type=float, default=60)
    ap.add_argument("--presidio-stop-after", type=float, default=10,
                    help="seconds: above this, larger sizes are not sent to Presidio")
    # The entities of ner.entity_weights in the gitops policy. Never add EMAIL_ADDRESS: its
    # recognizer stalls Presidio for about 30 s (see the policy comment).
    ap.add_argument("--presidio-entities", default="PERSON,LOCATION,NRP,MEDICAL_LICENSE,IBAN_CODE,"
                    "CREDIT_CARD,US_SSN,PHONE_NUMBER,IP_ADDRESS")
    ap.add_argument("--context-sizes", default="4000,16000,32000,64000,100000,128000")
    ap.add_argument("--parallel-size", type=int, default=16000,
                    help="tokens of each context of the parallel-context scenario")
    ap.add_argument("--parallel-levels", default="1,4,8")
    ap.add_argument("--policy", default="/app/litellm/privacy-plus.yaml",
                    help="privacy policy of the gate scenario (default: the LiteLLM pod's)")
    ap.add_argument("--gate-size", type=int, default=33000,
                    help="tokens of each agent context of the gate scenario")
    ap.add_argument("--gate-levels", default="1,4,8")
    ap.add_argument("--gate-rounds", type=int, default=2)
    ap.add_argument("--gate-threshold", type=float, default=0.70,
                    help="threshold of the gate scenario (0.70: the research tier of the agents)")
    ap.add_argument("--questions", default="",
                    help="JSON file with the systemone questions (default: the built-in ones)")
    ap.add_argument("--questions-label", default="",
                    help="name of the question set in the output (default: the file name)")
    args = ap.parse_args()
    global QUESTIONS_LABEL
    if args.questions:
        QUESTIONS_LABEL = args.questions_label or args.questions.rsplit("/", 1)[-1]
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
    if "context" in scen:
        scenario_context(b, args)
    if "parallel-context" in scen:
        scenario_parallel_context(b, args)
    if "gate" in scen:
        scenario_gate(args)


if __name__ == "__main__":
    main()
