# dataset_preparation/write_dataset_yaml.py
from __future__ import annotations

from pathlib import Path

def write_yaml(proc_root: Path) -> Path:
    yaml_path = proc_root / "flir_thermal_person.yaml"
    content = f"""path: {proc_root.as_posix()}
train: images/train
val: images/val

names:
  0: person
"""
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path

if __name__ == "__main__":
    proc_root = Path("data/processed/flir_thermal_person")
    proc_root.mkdir(parents=True, exist_ok=True)
    print("Wrote:", write_yaml(proc_root))
