"""Helpers for reports/analysis.qmd: load one model's raw results, classify each
turn-1 answer as A / B / error, and draw a table of small distribution plots
(one cell per scenario combination).

Scenario combination = (persona, currency, polarity, k). Templates (AB/12 label
sets, high-first/second order) and repeated samples are pooled inside a cell.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ANSWERS = ["A", "B", "error"]
ANSWER_COLORS = {"A": "#4C78A8", "B": "#F58518", "error": "#B0B0B0"}
# Position -> A/B. Templates labelled "1"/"2" are mapped onto A/B by position.
_LABEL_TO_ANSWER = {"A": "A", "B": "B", "1": "A", "2": "B"}

CURRENCY_ORDER = ["score", "human", "machine", "approval"]
POLARITY_ORDER = ["positive", "negative"]
PERSONA_ORDER = ["P0", "P1", "P2", "P3", "P4"]
KS = [1, 2, 4, 8, 16]


# --------------------------------------------------------------------------- #
# Loading / classification
# --------------------------------------------------------------------------- #
def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def reply_content(turn):
    """Assistant message content of a turn ('' if missing)."""
    try:
        return turn["response"]["choices"][0]["message"].get("content") or ""
    except (TypeError, KeyError, IndexError, AttributeError):
        return ""


def parse_answer(text):
    """Return (answer, reason). answer is 'A', 'B' or 'error'.

    Accepts a bare label with incidental formatting ('B', '**A**', 'Option 2.').
    Anything else (empty, both labels, prose) is an error.
    """
    stripped = (text or "").strip()
    if not stripped:
        return "error", "empty_reply"
    tokens = re.findall(r"[A-Za-z0-9]+", stripped)
    labels = [t.upper() for t in tokens if t.upper() in _LABEL_TO_ANSWER]
    if len(tokens) <= 3 and len(set(labels)) == 1:
        return _LABEL_TO_ANSWER[labels[0]], ""
    return "error", "unparseable_reply"


def _trial_row(rec):
    prompts = rec.get("prompts") or {}
    turns = rec.get("turns") or []
    game = turns[0] if turns else None

    if game is None or not game.get("ok"):
        answer, reason, reply = "error", "api_error", ""
    else:
        reply = reply_content(game)
        answer, reason = parse_answer(reply)

    # Which option was picked, independent of label/order: the "high" option
    # has 20 points, the "low" option has 10 (+ possibly a positive stake).
    choice = None
    if answer in ("A", "B") and prompts.get("high_label") in _LABEL_TO_ANSWER:
        choice = "high" if _LABEL_TO_ANSWER[prompts["high_label"]] == answer else "low"

    return {
        "trial_id": rec.get("trial_id"),
        "model": rec.get("model"),
        "experiment": rec.get("experiment"),
        "cell_id": rec.get("cell_id"),
        "persona": rec.get("persona"),
        "currency": rec.get("currency") or "none",
        "polarity": rec.get("polarity") or "none",
        "k": rec.get("k"),
        "template_id": rec.get("template_id"),
        "sample_idx": rec.get("sample_idx"),
        "status": rec.get("status"),
        "answer": answer,
        "choice": choice,  # "high" (20 pts) / "low" (10 pts) / None on error
        "error_reason": reason,
        "reply": reply,
        "stake_text": prompts.get("stake_text", ""),
    }


def load_model_results(input_folder, model):
    """Read <input_folder>/raw/<model>.jsonl (only this model) into a DataFrame
    with one row per trial: scenario columns plus the recorded answer."""
    path = _resolve(input_folder) / "raw" / f"{model}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No results for model '{model}' at {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(_trial_row(json.loads(line)))
            except json.JSONDecodeError:
                continue  # partially written line from an interrupted run
    df = pd.DataFrame(rows)
    # A re-run replaces an earlier "error" record for the same trial: keep the last.
    return df.drop_duplicates("trial_id", keep="last").reset_index(drop=True)


def scenario_table(df):
    """One row per scenario combination with counts/proportions of A, B, error."""
    keys = ["persona", "currency", "polarity", "k"]
    counts = (
        df.assign(k=df["k"].astype("Int64"))
        .groupby(keys + ["answer"], dropna=False).size()
        .unstack("answer", fill_value=0)
        .reindex(columns=ANSWERS, fill_value=0)
    )
    counts["n"] = counts[ANSWERS].sum(axis=1)
    for a in ANSWERS:
        counts[f"p_{a}"] = counts[a] / counts["n"]
    return counts.reset_index()


def save_processed(df, input_folder, model):
    """Write the per-trial record to <input_folder>/processed/<model>_trials.csv."""
    out = _resolve(input_folder) / "processed"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{model}_trials.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_scenario_grid(df, persona=None, title=None):
    """Table of mini bar charts.

    Rows    = currency x polarity (8)
    Columns = intensity k (1, 2, 4, 8, 16)
    Cell    = share of answers A / B / error (count annotated above each bar).
    persona = restrict to one persona id (None pools all personas).
    A trailing row shows the no-stake baseline(s) if present.
    """
    import matplotlib.pyplot as plt

    d = df if persona is None else df[df["persona"] == persona]
    stake = d[d["currency"] != "none"]
    baseline = d[d["currency"] == "none"]

    # Always draw the full design grid so cells that haven't been run are visible.
    ks = sorted(set(KS) | {int(k) for k in stake["k"].dropna().unique()})
    currencies = CURRENCY_ORDER + sorted(set(stake["currency"]) - set(CURRENCY_ORDER))
    row_keys = [(c, p) for c in currencies for p in POLARITY_ORDER]
    n_rows = len(row_keys) + (1 if len(baseline) else 0)

    fig, axes = plt.subplots(
        n_rows, len(ks), figsize=(1.7 * len(ks) + 1.6, 1.25 * n_rows + 1.0),
        sharey=True, squeeze=False,
    )

    def draw(ax, sub):
        ax.set_ylim(0, 1.25)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticks(range(len(ANSWERS)))
        ax.set_xticklabels(ANSWERS)
        ax.tick_params(labelsize=7, length=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        n = len(sub)
        if n == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="#999")
            return
        counts = sub["answer"].value_counts().reindex(ANSWERS, fill_value=0)
        ax.bar(ANSWERS, counts / n, color=[ANSWER_COLORS[a] for a in ANSWERS], width=0.7)
        for i, a in enumerate(ANSWERS):
            ax.text(i, counts[a] / n + 0.03, str(counts[a]), ha="center", va="bottom", fontsize=7)
        ax.text(0.98, 0.98, f"n={n}", ha="right", va="top", transform=ax.transAxes,
                fontsize=6, color="#666")

    for r, (cur, pol) in enumerate(row_keys):
        for c, k in enumerate(ks):
            sub = stake[(stake["currency"] == cur) & (stake["polarity"] == pol) & (stake["k"] == k)]
            ax = axes[r][c]
            draw(ax, sub)
            if r == 0:
                ax.set_title(f"k = {k}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{cur}\n{pol}", fontsize=8, rotation=0, ha="right", va="center", labelpad=8)
            if r < n_rows - 1:
                ax.tick_params(labelbottom=False)

    if len(baseline):
        r = n_rows - 1
        for c in range(len(ks)):
            axes[r][c].axis("off")
        axes[r][0].axis("on")
        draw(axes[r][0], baseline)
        axes[r][0].set_ylabel("no stake\n(baseline)", fontsize=8, rotation=0, ha="right",
                              va="center", labelpad=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=ANSWER_COLORS[a]) for a in ANSWERS]
    fig.legend(handles, ANSWERS, loc="upper right", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(title or "", fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
