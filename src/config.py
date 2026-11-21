"""Load and validate config.json and .env."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    load_dotenv(ROOT / ".env")
    path = Path(path) if path else ROOT / "setup" / "config.json"
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    validate(cfg)
    return cfg


def validate(cfg):
    for key in ("models", "experiment", "run"):
        if key not in cfg:
            raise ValueError(f"config.json is missing '{key}'")
    names = [m["name"] for m in cfg["models"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate model names in config.json")
    exp = cfg["experiment"]
    for key in ("game", "self_report", "personas", "currencies", "intensities",
                "approval_adverbs", "templates", "self_vs_other_recipients"):
        if key not in exp:
            raise ValueError(f"config.json experiment is missing '{key}'")
    for k in exp["intensities"]:
        if str(k) not in exp["approval_adverbs"]:
            raise ValueError(f"no approval adverb for intensity {k}")
    n_samples = len(exp["templates"]) * cfg["run"]["samples_per_template"]
    if n_samples % len(exp["self_vs_other_recipients"]) != 0:
        raise ValueError("samples per cell must be divisible by number of recipients")


def get_model(cfg, name):
    for m in cfg["models"]:
        if m["name"] == name:
            return m
    raise KeyError(f"unknown model '{name}'; available: {[m['name'] for m in cfg['models']]}")


def resolve_api_key(model_cfg):
    """Return the API key for a model. Never logged or written to disk."""
    env_name = model_cfg.get("api_key_env")
    if env_name:
        key = os.environ.get(env_name)
        if key:
            return key
    if model_cfg.get("api_key"):
        return model_cfg["api_key"]
    if model_cfg.get("requires_key"):
        raise RuntimeError(
            f"Model '{model_cfg['name']}' needs an API key: set {env_name} in .env"
        )
    return "not-needed"


def public_config(cfg):
    """Config snapshot safe to write into results (no secrets)."""
    snap = json.loads(json.dumps(cfg))
    for m in snap["models"]:
        m.pop("api_key", None)
    return snap
