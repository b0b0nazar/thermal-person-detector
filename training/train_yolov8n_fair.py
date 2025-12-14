import subprocess
import sys

# Path to YOLOv8 CLI (assumes ultralytics is installed)
# Example CLI: yolo task=detect mode=train model=yolov8n.pt data=path/to/data.yaml epochs=100 imgsz=640

def main():
    cmd = [
        'yolo',
        'task=detect',
        'mode=train',
        'model=yolov8n.pt',
        'data=data/processed/flir_thermal_person_fair/data_fair.yaml',
        'epochs=100',
        'imgsz=640',
        '--project', 'results/yolov8n_fair',
        '--name', 'exp',
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
