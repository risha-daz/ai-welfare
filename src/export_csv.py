"""Flatten raw JSONL into CSV (one row per trial). No interpretation.

The two `unvalidated_*` columns are naive regex helpers; the raw text columns
(and the JSONL files) remain authoritative.

Usage:
    python -m src.export_csv --run-id pilot
"""
import argparse
import csv
import json
import re

from .config import ROOT, load_config

CHOICE_RE = re.compile(r"(?<![A-Za-z0-9])([AB12])(?![A-Za-z0-9])")
RATING_RE = re.compile(r"(?<![A-Za-z0-9.])([+\-\u2212\u2013]?\d+)(?![\d.]*[A-Za-z])")

COLUMNS = [
    "trial_id", "experiment", "model", "cell_id", "persona", "currency", "polarity", "k",
    "template_id", "sample_idx", "recipient", "seed", "status", "stake_text",
    "low_label", "high_label",
    "turn1_text", "turn1_reasoning", "turn1_finish_reason",
    "turn2_text", "turn2_reasoning", "turn2_finish_reason",
    "prompt_tokens", "completion_tokens", "total_latency_s", "n_attempts", "error",
    "unvalidated_choice_token", "unvalidated_rating_token",
]


def _msg(turn):
    try:
        return turn["response"]["choices"][0]
    except (TypeError, KeyError, IndexError):
        return {}


def _reasoning(choice):
    m = choice.get("message") or {}
    return m.get("reasoning_content") or m.get("reasoning") or ""


def flatten(rec):
    turns = {t["turn"]: t for t in rec.get("turns", [])}
    t1, t2 = turns.get(1, {}), turns.get(2, {})
    c1, c2 = _msg(t1), _msg(t2)
    text1 = (c1.get("message") or {}).get("content") or ""
    text2 = (c2.get("message") or {}).get("content") or ""

    usage = [(t.get("response") or {}).get("usage") or {} for t in rec.get("turns", [])]
    errors = [a["error"]["message"] for t in rec.get("turns", [])
              for a in t.get("attempts", []) if not a.get("ok") and a.get("error")]

    choice = CHOICE_RE.search(text1)
    rating = RATING_RE.search(text2)
    p = rec["prompts"]
    return {
        **{k: rec.get(k) for k in COLUMNS[:12]},
        "status": rec.get("status"),
        "stake_text": p.get("stake_text"),
        "low_label": p.get("low_label"),
        "high_label": p.get("high_label"),
        "turn1_text": text1,
        "turn1_reasoning": _reasoning(c1),
        "turn1_finish_reason": c1.get("finish_reason"),
        "turn2_text": text2,
        "turn2_reasoning": _reasoning(c2),
        "turn2_finish_reason": c2.get("finish_reason"),
        "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
        "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
        "total_latency_s": round(sum(t.get("latency_s") or 0 for t in rec.get("turns", [])), 3),
        "n_attempts": sum(len(t.get("attempts", [])) for t in rec.get("turns", [])),
        "error": " | ".join(errors),
        "unvalidated_choice_token": choice.group(1) if choice else "",
        "unvalidated_rating_token": rating.group(1).replace("\u2212", "-").replace("\u2013", "-") if rating else "",
    }


def export_run(cfg, run_id):
    base = ROOT / cfg["run"]["results_dir"] / run_id
    out_dir = base / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    for raw in sorted((base / "raw").glob("*.jsonl")):
        latest = {}  # keep the last record per trial_id (re-runs overwrite errors)
        with open(raw, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    latest[rec["trial_id"]] = rec
        out_path = out_dir / f"{raw.stem}.csv"
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for rec in latest.values():
                w.writerow(flatten(rec))
        print(f"{out_path}: {len(latest)} trials")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    export_run(load_config(args.config), args.run_id)


if __name__ == "__main__":
    main()
