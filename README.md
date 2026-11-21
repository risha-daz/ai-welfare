# Welfare in AI Systems

## Introduction

Have you ever wondered whether to tried to treat AI systems as sentient entities? Maybe in a one-on-one chat interface? What about when it is a repetitive "mechanical" task, like an automated repeated prompt? I have also wondered whether this has an effect on the responses and whether one or the other is a better approach.

In short "welfare subjects" are entities which can be said to possess something akin to a "conciousness" and further, the ability to have "positive" or "negative" experiences (referred to as "valence" henceforth). Then, an ability to possess "agency", where any actions that entity makes has an intention which takes into account this sense of self. Such as Even with living beings, this is not a fully understood, so it is a hotly debated topic when it comes to artificial systems.

> This is in part inspired by a take-home-assessment inititated by Anthropic for an internship program. I wrote this proposal and got curious of its results so i decided to run this myself and see the results.

Please head over to <proposal.md> for the full breakdown.

## Setup

Prerequisites: 
- python 3.10+
- lmstudio (for local llm) or api key (for llm api)

#### Clone the repository

```bash
git clone <this repo>
cd ai-welfare
pip install -r requirements.txt
```

#### Setup your LLM

Models are OpenAI-compatible endpoints listed under `models` in `config.json`. For hosted APIs, put the key in `.env` (git-ignored) under the variable named by `api_key_env`, e.g. `GROQ_API_KEY=...`. For LM Studio, start the local server (default `http://localhost:1234/v1`) and load the model.

#### Configure your scenarios and personas

Personas, stake currencies, intensities, prompt templates and run settings (temperature, samples per cell, seed, `max_tokens`) are all in `config.json` and follow `method.md` (A2).

#### Run multiple iterations

```bash
# 1. preview trial counts and a sample prompt, no API calls
python -m src.run --experiment pilot --model all --dry-run

# 2. small smoke test (results/smoke/)
python -m src.run --experiment pilot --model all --limit 5 --run-id smoke

# 3. real runs: pilot (10% of cells, n=5), main grid (4,100 trials/model), self-vs-other control (800/model)
python -m src.run --experiment pilot --model all
python -m src.run --experiment main --model all
python -m src.run --experiment self_vs_other --model all

# 4. flatten raw results to CSV
python -m src.export_csv --run-id main
```

Runs are resumable: re-run the same command and finished trials are skipped (trials that ended in an error are retried). Use `--limit N` to run in chunks.

## Repository Structure

```
setup/config.json        models, prompts, personas, stakes, run settings
.env               API keys (git-ignored)
src/
  config.py        load/validate config.json and .env
  prompts.py       prompt construction (method.md A2)
  grid.py          deterministic trial generation: main, self_vs_other, pilot
  client.py        OpenAI-compatible chat call with logged retries
  run.py           CLI runner (resumable, concurrent, writes raw JSONL)
  export_csv.py    flatten raw JSONL to CSV (no interpretation)
results/<run_id>/  (git-ignored)
  manifest.json    config snapshot (no secrets), git commit, invocations
  raw/<model>.jsonl  one line per trial: exact prompts, full API responses, retry log
  csv/<model>.csv    flattened export
```

## Workflow

The script runs each trial as a two-turn conversation (the game, then the self-report) on every selected model and appends the full raw exchange to `results/<run_id>/raw/<model>.jsonl`: exact messages, the complete API response (including any reasoning field, finish reason and token usage), latencies, and every retry or error. Nothing is parsed or filtered at run time; exclusion rules from `method.md` A4 are applied later. `export_csv.py` flattens this to CSV and adds two naive, unvalidated helper columns (first A/B/1/2 token and first integer in the self-report). Analysis, plots, J-lens and the Assistant Axis are not part of this step yet; the endpoints used here do not expose the residual stream.

## Cost

Please use a cost calculator to check what api costs might incur. 

## Results

<link>


