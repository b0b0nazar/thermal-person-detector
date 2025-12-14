"""
Visualize 3–5 sample images with bounding boxes from the processed YOLO dataset.
"""
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random

# Class names (adjust as needed)
CLASS_NAMES = ["person", "bicycle", "car", "dog"]

def draw_yolo_boxes(img_path, label_path, class_names):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if not label_path.exists():
        return img
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts)
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = class_names[int(cls)] if int(cls) < len(class_names) else str(int(cls))
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

def main(proc_root, split="train", num_samples=5):
    img_dir = Path(proc_root) / "images" / split
    lbl_dir = Path(proc_root) / "labels" / split
    img_files = list(img_dir.glob("*.jpeg"))
    if len(img_files) == 0:
        print("No images found.")
        return
    samples = random.sample(img_files, min(num_samples, len(img_files)))
    plt.figure(figsize=(15, 3*len(samples)))
    for i, img_path in enumerate(samples):
        label_path = lbl_dir / (img_path.stem + ".txt")
        img = draw_yolo_boxes(img_path, label_path, CLASS_NAMES)
        plt.subplot(len(samples), 1, i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(img_path.name)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_root", type=str, default="data/processed/flir_thermal_person", help="Processed dataset root")
    parser.add_argument("--split", type=str, default="train", help="Dataset split: train or val")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    args = parser.parse_args()
    main(args.proc_root, args.split, args.num_samples)