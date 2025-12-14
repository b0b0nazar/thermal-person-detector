import cv2
import glob
import os
from ultralytics import YOLO

# Paths
MODEL_PATH = 'results/flir_person_yolov8n_fair/weights/best.pt'
IMG_DIR = 'data/raw/flir_adas/video/thermal_8_bit/'
VIDEO_OUTPUT = 'results/flir_person_yolov8n_fair/qualitative_val/video_inference_demo_from_images.mp4'

# Load model
model = YOLO(MODEL_PATH)

# Get sorted list of image files
img_files = sorted(glob.glob(os.path.join(IMG_DIR, '*.jpeg')) + glob.glob(os.path.join(IMG_DIR, '*.jpg')) + glob.glob(os.path.join(IMG_DIR, '*.png')))
if not img_files:
    print(f'No images found in {IMG_DIR}')
    exit(1)

# Read first image to get size
first_img = cv2.imread(img_files[0])
height, width = first_img.shape[:2]

# Output video writer
os.makedirs(os.path.dirname(VIDEO_OUTPUT), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 15  # Set FPS for output video
out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

for img_path in img_files:
    frame = cv2.imread(img_path)
    if frame is None:
        continue
    # Run YOLO inference
    results = model(frame)
    # Draw boxes
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else []
        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else []
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'person {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    # Show frame
    cv2.imshow('YOLOv8 Image Sequence Inference', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
print(f'Inference complete. Output saved to {VIDEO_OUTPUT}')
