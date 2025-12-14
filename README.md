# Thermal Person Detection with YOLOv8n and RT-DETR

## Overview
This project focuses on detecting persons in thermal images using two state-of-the-art object detection models: YOLOv8n (Ultralytics) and RT-DETR (Baidu). The pipeline covers dataset preparation, model training, evaluation, and qualitative/quantitative analysis, with a strong emphasis on reproducibility and code clarity.

## Project Structure
```
data/
  processed/
    flir_thermal_person_fair/           # Fairly split dataset and YOLO labels
  raw/
    flir_adas/
      train/val/video/                  # Original FLIR ADAS images and COCO annotations
      ...
dataset_preparation/                    # Scripts for data splitting, conversion, and audits
inference_demo/                         # Scripts for running inference on images and video sequences
results/
  flir_person_yolov8n_fair/             # YOLOv8n training results, metrics, and visualizations
  flir_person_rtdetr-l/                 # RT-DETR training results, metrics, and visualizations
  qualitative_examples/                 # Improved qualitative results for presentation
training/                               # Training and visualization scripts for both models
```

## Dataset Preparation
- **Source:** FLIR ADAS thermal dataset (see `data/raw/flir_adas/`)
- **Splitting:** Stratified and fair splitting using `dataset_preparation/stratified_split_yolo.py`
- **Annotation Conversion:** COCO to YOLO format using custom scripts in `dataset_preparation/`
- **Audit:** Class balance and split integrity checked with `dataset_preparation/check_yolo_split_balance.py`

## Model Training
- **YOLOv8n:**
  - Training script: `training/train_yolov8n_fair.py`
  - Config: `data/processed/flir_thermal_person_fair/data_fair.yaml`
  - Image size: 640x640
  - Training time: ~45 minutes for 50 epochs
  - Results: `results/flir_person_yolov8n_fair/`
- **RT-DETR:**
  - Training script: `training/train_rtdetr_fair.py`
  - Config: `data/processed/flir_thermal_person_fair/data_fair.yaml`
  - Results: `results/flir_person_rtdetr-l/`

## Inference & Evaluation
- **Image and Video Inference:**
  - Scripts: `inference_demo/image_sequence_inference_demo.py` (YOLOv8n), `inference_demo/image_sequence_inference_rtdetr.py` (RT-DETR)
  - Outputs: Annotated images and videos in `results/`
- **Qualitative Analysis:**
  - Improved visualizations in `results/qualitative_examples/`
  - Includes both successful detections and failure cases (missed detections, false alarms)
- **Quantitative Metrics:**
  - mAP, precision, recall, and confusion matrices in `results/flir_person_yolov8n_fair/` and `results/flir_person_rtdetr-l/`

## Reproducibility
- All scripts are modular and documented.
- Data splits and label conversions are versioned and auditable.
- Training and inference can be reproduced using provided scripts and config files.

## How to Run
1. **Prepare the dataset:**
   - Place raw FLIR ADAS data in `data/raw/flir_adas/`
   - Run data split and conversion scripts in `dataset_preparation/`
2. **Train models:**
   - `python3 training/train_yolov8n_fair.py`
   - `python3 training/train_rtdetr_fair.py`
3. **Run inference:**
   - `python3 inference_demo/image_sequence_inference_demo.py`
   - `python3 inference_demo/image_sequence_inference_rtdetr.py`
4. **Visualize results:**
   - Check `results/qualitative_examples/` and other results folders

## References
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [RT-DETR GitHub](https://github.com/ModelTC/RT-DETR)
- FLIR ADAS Dataset

## Contact
For questions or collaboration, please open an issue or contact the project maintainer.
