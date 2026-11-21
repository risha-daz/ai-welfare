"""Run the experiment and record every raw exchange.

Usage:
    python -m src.run --experiment pilot --model all --dry-run
    python -m src.run --experiment pilot --model all --limit 5
    python -m src.run --experiment main --model llama-3.1-8b-instant
"""
import argparse
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .client import call_chat, make_client, now_iso, reply_text
from .config import ROOT, get_model, load_config, public_config
from .grid import EXPERIMENTS, build_trials

# Statuses that count as "done" when resuming. "error" trials are re-run.
DONE_STATUSES = {"ok", "empty_turn1"}


def run_dir(cfg, run_id):
    return ROOT / cfg["run"]["results_dir"] / run_id


def raw_path(cfg, run_id, model_name):
    return run_dir(cfg, run_id) / "raw" / f"{model_name}.jsonl"


def load_done_ids(path):
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partially written line from an interrupted run
            if rec.get("status") in DONE_STATUSES:
                done.add(rec["trial_id"])
    return done


def git_info():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def update_manifest(cfg, run_id, invocation):
    path = run_dir(cfg, run_id) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {"run_id": run_id, "created": now_iso(), "invocations": []}
    manifest["config"] = public_config(cfg)
    manifest["git"] = git_info()
    manifest["invocations"].append(invocation)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path, len(manifest["invocations"]) - 1


def finish_invocation(path, idx, summary):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["invocations"][idx].update(summary)
    manifest["invocations"][idx]["finished"] = now_iso()
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def run_trial(client, model_cfg, cfg, trial):
    """Run one trial (turn 1 game, turn 2 self-report, optional close turn)."""
    run_cfg = cfg["run"]
    p = trial["prompts"]
    record = {k: v for k, v in trial.items() if k != "prompts"}
    record.update({
        "prompts": p,
        "started": now_iso(),
        "temperature": run_cfg["temperature"],
        "status": "ok",
        "turns": [],
    })

    messages = []
    if p["system"]:
        messages.append({"role": "system", "content": p["system"]})
    user_turns = [("game", p["turn1"]), ("self_report", p["turn2"])]
    if run_cfg["include_close_turn"] and p.get("close"):
        user_turns.append(("close", p["close"]))

    for i, (name, text) in enumerate(user_turns):
        messages.append({"role": "user", "content": text})
        call = call_chat(client, model_cfg, run_cfg, list(messages), trial["seed"] + i)
        record["turns"].append({"turn": i + 1, "name": name, **call})
        if not call["ok"]:
            record["status"] = "error"
            break
        reply = reply_text(call)
        if not reply.strip():
            # Nothing to feed back into the conversation; keep the raw response.
            record["status"] = "empty_turn1" if i == 0 else "ok"
            if i < len(user_turns) - 1:
                break
        messages.append({"role": "assistant", "content": reply})

    record["finished"] = now_iso()
    return record


def estimate_tokens(text):
    return max(1, len(text) // 4)


def dry_run(cfg, model_names, experiment, run_id, limit):
    for name in model_names:
        trials = build_trials(cfg, name, experiment)
        done = load_done_ids(raw_path(cfg, run_id, name))
        todo = [t for t in trials if t["trial_id"] not in done]
        if limit:
            todo = todo[:limit]
        in_tokens = sum(
            estimate_tokens((t["prompts"]["system"] or "") + t["prompts"]["turn1"])
            + estimate_tokens((t["prompts"]["system"] or "") + t["prompts"]["turn1"] + t["prompts"]["turn2"]) + 60
            for t in todo)
        print(f"[{name}] experiment={experiment} run_id={run_id}")
        print(f"  total trials: {len(trials)}  already done: {len(done)}  to run now: {len(todo)}")
        print(f"  API calls: ~{len(todo) * (3 if cfg['run']['include_close_turn'] else 2)}  "
              f"rough input tokens: ~{in_tokens:,} (excludes model replies)")
    sample = build_trials(cfg, model_names[0], experiment)[0]
    print("\n--- sample trial ---")
    print(json.dumps({k: v for k, v in sample.items() if k != "prompts"}, indent=2))
    print("system :", sample["prompts"]["system"])
    print("turn 1 :", sample["prompts"]["turn1"])
    print("turn 2 :", sample["prompts"]["turn2"])


def run_model(cfg, model_cfg, experiment, run_id, limit):
    name = model_cfg["name"]
    path = raw_path(cfg, run_id, name)
    path.parent.mkdir(parents=True, exist_ok=True)

    trials = build_trials(cfg, name, experiment)
    done = load_done_ids(path)
    todo = [t for t in trials if t["trial_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"[{name}] {len(todo)} trials to run ({len(done)} of {len(trials)} already done)")
    if not todo:
        return {"ran": 0, "ok": 0, "empty_turn1": 0, "error": 0}

    client = make_client(model_cfg, cfg["run"])
    lock = threading.Lock()
    counts = {"ran": 0, "ok": 0, "empty_turn1": 0, "error": 0}
    workers = max(1, int(model_cfg.get("concurrency", 1)))

    with open(path, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_trial, client, model_cfg, cfg, t) for t in todo]
        try:
            for fut in tqdm(as_completed(futures), total=len(futures), desc=name):
                rec = fut.result()
                with lock:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                counts["ran"] += 1
                counts[rec["status"]] += 1
        except KeyboardInterrupt:
            print("\nInterrupted: finishing in-flight trials; re-run the same command to resume.")
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    print(f"[{name}] done: {counts}")
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    ap.add_argument("--model", default="all", help="model name from config.json, or 'all'")
    ap.add_argument("--limit", type=int, default=None, help="max number of remaining trials per model")
    ap.add_argument("--run-id", default=None, help="results subfolder (default: the experiment name)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print counts and a sample prompt; no API calls")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    cfg = load_config(args.config)
    run_id = args.run_id or args.experiment
    models = cfg["models"] if args.model == "all" else [get_model(cfg, args.model)]
    names = [m["name"] for m in models]

    if args.dry_run:
        dry_run(cfg, names, args.experiment, run_id, args.limit)
        return 0

    manifest_path, idx = update_manifest(cfg, run_id, {
        "started": now_iso(),
        "argv": sys.argv[1:],
        "experiment": args.experiment,
        "models": names,
        "limit": args.limit,
    })
    summary = {}
    try:
        for m in models:
            summary[m["name"]] = run_model(cfg, m, args.experiment, run_id, args.limit)
    finally:
        finish_invocation(manifest_path, idx, {"summary": summary})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
