"""Canonical class list shared by both tracks. Must match datasets/classes.json."""
import json
from pathlib import Path

_CLASSES_JSON = Path(__file__).resolve().parents[2] / "datasets" / "classes.json"

CLASS_NAMES = json.loads(_CLASSES_JSON.read_text())["classes"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
