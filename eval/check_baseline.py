#!/usr/bin/env python3
"""Run the evaluation sets and compare the results with eval/baseline.json.

run_eval.py (copied unchanged from the old repo) exits with 1 on any leak. The sets
contain on purpose some cases that fail today, so that exit code cannot gate the CI.
This script runs run_eval.py for every set, reads the per-case CSV and fails only on
a NEW leak: a case that should stay LOCAL, went to SOTA, and is not in the baseline.

Usage:
  uv run eval/check_baseline.py            # compare, exit 1 on a new leak
  uv run eval/check_baseline.py --update   # write the current results as the baseline

Env: PRESIDIO_URL (default http://localhost:3000/analyze), LID_MODEL_PATH
(default .cache/lid.176.ftz, see scripts/fetch-lid-model.sh), POLICY_DIR (default
tests/policy). The C2 classifier stays off: the CI has no LLM.
"""

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "eval" / "baseline.json"
LID_MODEL = ROOT / ".cache" / "lid.176.ftz"

# name -> (policy file, evaluation set)
SETS = {
    "english": ("privacy-plus.yaml", "privacy-plus-english.yaml"),
    "italian": ("privacy-plus.yaml", "privacy-plus-italian.yaml"),
    "chain-demo": ("chain.yaml", "chain-demo.yaml"),
}


def run_set(name, policy_dir):
    policy, cases = SETS[name]
    with tempfile.TemporaryDirectory() as tmp:
        csv_out = pathlib.Path(tmp) / f"{name}.csv"
        env = dict(os.environ)
        env.update({
            "PYTHONPATH": str(ROOT / "litellm"),
            "POLICY_FILE": str(policy_dir / policy),
            "EVAL_FILE": str(ROOT / "eval" / cases),
            "PRESIDIO_URL": os.environ.get("PRESIDIO_URL", "http://localhost:3000/analyze"),
            "LID_MODEL_PATH": os.environ.get("LID_MODEL_PATH", str(LID_MODEL)),
            "CSV_OUT": str(csv_out),
            "EVAL_LABEL": name,
        })
        for var in ("CLASSIFIER_ENABLED", "CLASSIFIER_GRAY_LOW"):
            env.pop(var, None)
        proc = subprocess.run([sys.executable, str(ROOT / "eval" / "run_eval.py")], env=env,
                              capture_output=True, text=True)
        if not csv_out.exists():
            sys.stderr.write(proc.stdout + proc.stderr)
            sys.exit(f"{name}: run_eval.py wrote no results (exit {proc.returncode})")
        rows = list(csv.DictReader(csv_out.open()))
    errors = [r["id"] for r in rows if "error" in r["signals"]]
    if errors:
        sys.exit(f"{name}: backend errors on {len(errors)} cases (is Presidio up?): {errors[:5]}")
    leaks = sorted(r["id"] for r in rows if r["expect"] == "local" and r["decision"] == "sota")
    false_pos = sorted(r["id"] for r in rows if r["expect"] == "sota" and r["decision"] == "local")
    correct = sum(r["correct"] == "1" for r in rows)
    return {"cases": len(rows), "correct": correct, "leaks": leaks, "false_positives": false_pos}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true", help="write the baseline")
    parser.add_argument("--sets", nargs="*", default=list(SETS), choices=list(SETS))
    args = parser.parse_args()
    policy_dir = pathlib.Path(os.environ.get("POLICY_DIR", ROOT / "tests" / "policy"))

    results = {name: run_set(name, policy_dir) for name in args.sets}
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}

    failed = False
    for name, res in results.items():
        base = baseline.get(name, {"leaks": [], "false_positives": []})
        new_leaks = sorted(set(res["leaks"]) - set(base["leaks"]))
        fixed_leaks = sorted(set(base["leaks"]) - set(res["leaks"]))
        new_fp = sorted(set(res["false_positives"]) - set(base["false_positives"]))
        fixed_fp = sorted(set(base["false_positives"]) - set(res["false_positives"]))
        print(f"{name}: {res['correct']}/{res['cases']} correct, "
              f"{len(res['leaks'])} leaks, {len(res['false_positives'])} false positives")
        for label, ids in (("NEW LEAKS", new_leaks), ("fixed leaks", fixed_leaks),
                           ("new false positives", new_fp), ("fixed false positives", fixed_fp)):
            if ids:
                print(f"  {label}: {', '.join(ids)}")
        failed |= bool(new_leaks) and not args.update

    if args.update:
        BASELINE.write_text(json.dumps({**baseline, **results}, indent=2) + "\n")
        print(f"wrote {BASELINE.relative_to(ROOT)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
