"""
Script to count positive (images with at least one person) and negative (no person) images in YOLO label folders.
Prints counts and ratios for train and val splits.
 python3 dataset_preparation/check_yolo_split_balance.py
"""
from pathlib import Path

def count_pos_neg(label_dir: Path):
    pos = 0
    neg = 0
    for lbl_file in label_dir.glob("*.txt"):
        if lbl_file.stat().st_size == 0:
            neg += 1
        else:
            pos += 1
    total = pos + neg
    ratio = pos / total if total > 0 else 0
    return pos, neg, total, ratio

def main(proc_dir: Path = Path("data/processed/flir_thermal_person_fair")):
    for split in ["train", "val"]:
        label_dir = proc_dir / "labels" / split
        pos, neg, total, ratio = count_pos_neg(label_dir)
        print(f"--- {split.capitalize()} Split ---")
        print(f"Total images: {total}")
        print(f"  Positive (with person): {pos} ({ratio:.2%})")
        print(f"  Negative (no person):   {neg} ({(1-ratio):.2%})\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_dir", type=Path, default=Path("data/processed/flir_thermal_person_fair"),
                        help="Processed dataset root directory")
    args = parser.parse_args()
    main(args.proc_dir)
