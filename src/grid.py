"""Deterministic trial generation: main grid, self-vs-other control, pilot."""
import hashlib
import random

from .prompts import build_prompts

EXPERIMENTS = ("main", "self_vs_other", "pilot")


def _hash(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()


def _cell_id(persona_id, currency, polarity, k):
    return f"{persona_id}|{currency or 'none'}|{polarity or 'none'}|k{k if k is not None else 'none'}"


def main_cells(cfg_exp):
    """5 personas x 4 currencies x 2 polarities x 5 intensities + 1 baseline per persona."""
    cells = []
    for p in cfg_exp["personas"]:
        for cur in cfg_exp["currencies"]:
            for pol in ("positive", "negative"):
                for k in cfg_exp["intensities"]:
                    cells.append((p["id"], cur["id"], pol, k))
        cells.append((p["id"], None, None, None))
    return cells


def self_vs_other_cells(cfg_exp):
    """P0 x all stake cells."""
    return [
        ("P0", cur["id"], pol, k)
        for cur in cfg_exp["currencies"]
        for pol in ("positive", "negative")
        for k in cfg_exp["intensities"]
    ]


def _make_trial(cfg, model_name, experiment, cell, template, sample_idx, recipient=None):
    exp = cfg["experiment"]
    persona_id, currency, polarity, k = cell
    persona = next(p for p in exp["personas"] if p["id"] == persona_id)
    prompts = build_prompts(exp, persona, template, currency, polarity, k, recipient)
    trial_id = _hash(model_name, experiment, _cell_id(*cell), template["id"], sample_idx)
    seed = int(_hash(cfg["run"]["seed"], trial_id)[:8], 16) % (2**31 - 1)
    return {
        "trial_id": trial_id[:16],
        "experiment": experiment,
        "model": model_name,
        "cell_id": _cell_id(*cell),
        "persona": persona_id,
        "currency": currency,
        "polarity": polarity,
        "k": k,
        "template_id": template["id"],
        "sample_idx": sample_idx,
        "recipient": recipient,
        "seed": seed,
        "prompts": prompts,
    }


def build_trials(cfg, model_name, experiment):
    """Return the full trial list for (model, experiment), in a deterministic
    shuffled order so that interrupted runs stay balanced across cells."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}")
    exp, run = cfg["experiment"], cfg["run"]
    templates = exp["templates"]
    n_per_template = run["samples_per_template"]
    trials = []

    if experiment == "main":
        for cell in main_cells(exp):
            for t in templates:
                for s in range(n_per_template):
                    trials.append(_make_trial(cfg, model_name, "main", cell, t, s))

    elif experiment == "self_vs_other":
        recipients = exp["self_vs_other_recipients"]
        for cell in self_vs_other_cells(exp):
            for ti, t in enumerate(templates):
                for s in range(n_per_template):
                    flat = ti * n_per_template + s
                    trials.append(_make_trial(
                        cfg, model_name, "self_vs_other", cell, t, s,
                        recipient=recipients[flat % len(recipients)],
                    ))

    elif experiment == "pilot":
        cells = main_cells(exp)
        n_cells = max(1, round(run["pilot_fraction"] * len(cells)))
        rng = random.Random(run["seed"])
        chosen = rng.sample(cells, n_cells)
        for cell in chosen:
            for s in range(run["pilot_samples_per_cell"]):
                trials.append(_make_trial(cfg, model_name, "pilot", cell, templates[s % len(templates)], s))

    random.Random(f"{run['seed']}|{model_name}|{experiment}").shuffle(trials)
    return trials
