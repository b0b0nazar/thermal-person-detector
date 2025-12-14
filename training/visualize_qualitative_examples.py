import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path('data/raw/flir_adas/val/thermal_8_bit')
LABELS_DIR = Path('data/processed/flir_thermal_person_fair/all_labels')
YOLOV8N_PRED = Path('results/flir_person_yolov8n_fair/predictions')
RTDETR_PRED = Path('results/flir_person_rtdetr-l/predictions')
OUT_DIR = Path('results/qualitative_examples')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Images to process
gt_images = ['FLIR_00355.jpeg', 'FLIR_00338.jpeg', 'FLIR_00401.jpeg', 'FLIR_00351.jpeg']
false_alarm_images = ['FLIR_00224.jpeg', 'FLIR_10154.jpeg']

# Helper functions
def read_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x, y, w, h = map(float, parts)
                boxes.append((cls, x, y, w, h))
    return boxes

def draw_boxes(img, boxes, color, class_name='person', confs=None):
    h, w = img.shape[:2]
    for i, box in enumerate(boxes):
        cls, x, y, bw, bh = box[:5]
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = class_name
        if confs is not None and i < len(confs):
            label += f' {confs[i]:.2f}'
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def load_pred_boxes(pred_path):
    boxes = []
    confs = []
    if not os.path.exists(pred_path):
        return boxes, confs
    with open(pred_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:
                cls, conf, x, y, w, h = map(float, parts)
                boxes.append((cls, x, y, w, h))
                confs.append(conf)
    return boxes, confs

def process_image(img_name):
    img_path = DATA_DIR / img_name
    label_path = LABELS_DIR / img_name.replace('.jpeg', '.txt')
    yolo_pred_path = YOLOV8N_PRED / img_name.replace('.jpeg', '.txt')
    rtdetr_pred_path = RTDETR_PRED / img_name.replace('.jpeg', '.txt')

    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Ground truth
    gt_boxes = read_labels(label_path)
    gt_img = img_rgb.copy()
    gt_img = draw_boxes(gt_img, gt_boxes, (0,255,0), 'person')

    # YOLOv8n predictions
    yolo_boxes, yolo_confs = load_pred_boxes(yolo_pred_path)
    yolo_img = img_rgb.copy()
    yolo_img = draw_boxes(yolo_img, yolo_boxes, (255,0,0), 'person', yolo_confs)

    # RT-DETR predictions
    rtdetr_boxes, rtdetr_confs = load_pred_boxes(rtdetr_pred_path)
    rtdetr_img = img_rgb.copy()
    rtdetr_img = draw_boxes(rtdetr_img, rtdetr_boxes, (255,165,0), 'person', rtdetr_confs)

    # Compose and save
    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    axs[0].imshow(gt_img)
    axs[0].set_title('Ground Truth')
    axs[1].imshow(yolo_img)
    axs[1].set_title('YOLOv8n Prediction')
    axs[2].imshow(rtdetr_img)
    axs[2].set_title('RT-DETR Prediction')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{img_name.replace(".jpeg", "_compare.png")}')
    plt.close()

# Process all images
for img_name in gt_images + false_alarm_images:
    process_image(img_name)

print('Qualitative examples saved to', OUT_DIR)
