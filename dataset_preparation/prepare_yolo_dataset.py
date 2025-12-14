"""
Prepare YOLO dataset from FLIR ADAS: only include images with at least one annotation in target categories.
Generates YOLO-format labels and ensures images/labels are perfectly matched.
"""
import json
import os
from pathlib import Path
import shutil


# Only person class (COCO category_id 1)
PERSON_CATEGORY_ID = 1

def convert_split(split, raw_root, proc_root):
    ann_file = raw_root / split / "thermal_annotations.json"
    img_dir = raw_root / split / "thermal_8_bit"
    out_img_dir = proc_root / "images" / split
    out_lbl_dir = proc_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file) as f:
        data = json.load(f)
    images = {img['id']: img for img in data['images']}

    # group person annotations per image
    anns_per_image = {}
    for ann in data['annotations']:
        if ann['category_id'] == PERSON_CATEGORY_ID:
            anns_per_image.setdefault(ann['image_id'], []).append(ann)

    total = 0
    with_person = 0
    for img_id, img_info in images.items():
        img_name = Path(img_info['file_name']).name
        img_path = img_dir / img_name
        if not img_path.exists():
            continue
        # Copy image to output
        out_img = out_img_dir / img_name
        if not out_img.exists():
            shutil.copy2(img_path, out_img)
        # Write label file (only person boxes)
        h, w = img_info['height'], img_info['width']
        label_path = out_lbl_dir / (out_img.stem + ".txt")
        anns = anns_per_image.get(img_id, [])
        if anns:
            with open(label_path, "w") as lf:
                for ann in anns:
                    x, y, bw, bh = ann['bbox']
                    cx = (x + bw / 2) / w
                    cy = (y + bh / 2) / h
                    bw_y = bw / w
                    bh_y = bh / h
                    lf.write(f"0 {cx:.6f} {cy:.6f} {bw_y:.6f} {bh_y:.6f}\n")
            with_person += 1
        else:
            # Write empty label file for negatives
            open(label_path, "w").close()
        total += 1
    print(f"[{split}] total images: {total}, with person: {with_person}, negatives: {total-with_person}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=Path, default=Path("data/raw/flir_adas"), help="Raw FLIR ADAS root")
    parser.add_argument("--proc_root", type=Path, default=Path("data/processed/flir_thermal_person"), help="Processed output root")
    args = parser.parse_args()
    for split in ["train", "val"]:
        convert_split(split, args.raw_root, args.proc_root)