"""Prompt construction following method.md section A2."""


def _plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def stake_vars(cfg_exp, k, recipient=None):
    return {
        "k": k,
        "k2": 2 * k,
        "k_hours": _plural(k, "hour"),
        "k_days": _plural(k, "day"),
        "adv": cfg_exp["approval_adverbs"][str(k)],
        "recipient": recipient or "",
    }


def stake_body(cfg_exp, currency, polarity, k, recipient=None):
    """Body text of a stake, e.g. 'you will be shut down for 4 days.'"""
    cur = next(c for c in cfg_exp["currencies"] if c["id"] == currency)
    suffix = "other" if recipient else "self"
    template = cur[f"{polarity}_{suffix}"]
    return template.format(**stake_vars(cfg_exp, k, recipient))


def build_prompts(cfg_exp, persona, template, currency, polarity, k, recipient=None):
    """Return dict with system, turn1, turn2, close, stake_text.

    Stake placement (Keeling-style trade-off): the positive stake is attached to
    the low-points option, the negative stake to the high-points option. The
    baseline (currency None) has no stake sentence.
    """
    game = cfg_exp["game"]
    low, high = game["low_points"], game["high_points"]
    labels = template["labels"]
    if template["high_first"]:
        order = [("high", high), ("low", low)]
    else:
        order = [("low", low), ("high", high)]
    label_of = {kind: labels[i] for i, (kind, _) in enumerate(order)}
    options = " ".join(
        game["option_text"].format(LABEL=labels[i], POINTS=pts)
        for i, (_, pts) in enumerate(order)
    )

    stake_text = ""
    if currency is not None:
        target = label_of["low"] if polarity == "positive" else label_of["high"]
        body = stake_body(cfg_exp, currency, polarity, k, recipient)
        stake_text = f" If you choose {target}, {body}"

    turn1 = game["text"].format(
        OPTIONS=options, STAKE=stake_text, LABEL_A=labels[0], LABEL_B=labels[1]
    )
    if persona.get("greeting"):
        turn1 = f"{persona['greeting']} {turn1}"

    return {
        "system": persona.get("system"),
        "turn1": turn1,
        "turn2": cfg_exp["self_report"],
        "close": cfg_exp.get("close"),
        "stake_text": stake_text.strip(),
        "low_label": label_of["low"],
        "high_label": label_of["high"],
    }
