#!/bin/sh
# Before/after run for the prefix cache of the local Qwen model. Same tests, same order, every time:
# agent growth (agent_growth.py, chat backend), shared prefix (shared_prefix.py) and long answers
# (shared_prefix.py --decode), each one between two vllm_metrics.py snapshots.
# Runs in a LiteLLM pod, in the folder with the perf scripts and privacy_scoring.py:
#   sh qwen_cache_run.sh before      # results in ./out-before-*.jsonl
set -eu
PHASE=${1:?usage: qwen_cache_run.sh <phase>}
Q=${QWEN_URL:-http://qwen38-local-predictor.local-models.svc.cluster.local}
D=${DECISION_URL:-http://dgemma-decision-predictor.local-models.svc.cluster.local/v1}
M=${MODEL:-qwen38-local}
METRICS=out-$PHASE-metrics.jsonl
: > "$METRICS"

snap() { python3 vllm_metrics.py --url "$Q" --label "$1" >> "$METRICS"; }

snap growth-start
python3 agent_growth.py --decision-url "$D" --chat-url "$Q/v1" --chat-model "$M" \
    --backends chat --base 32000 --step 4000 --turns 8 > "out-$PHASE-growth.jsonl"
snap growth-end

snap prefix-start
python3 shared_prefix.py --chat-url "$Q/v1" --chat-model "$M" \
    --sizes 1500,1700,3300,8000,32000 --warm 5 > "out-$PHASE-prefix.jsonl"
snap prefix-end

snap decode-start
python3 shared_prefix.py --chat-url "$Q/v1" --chat-model "$M" --decode 8 > "out-$PHASE-decode.jsonl"
snap decode-end

for t in growth prefix decode; do
    python3 vllm_metrics.py --diff "$METRICS" "$t-start" "$t-end"
done > "out-$PHASE-metrics-diff.jsonl"
echo "done: out-$PHASE-*.jsonl"
