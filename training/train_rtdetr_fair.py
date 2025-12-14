import subprocess
import sys

# Path to RT-DETR CLI (assumes rtdetr package or script is available)
# Example CLI: python rtdetr/train.py --data data.yaml --model rtdetr-l.yaml --epochs 100 --imgsz 640

def main():
    cmd = [
        'python3', 'rtdetr/train.py',
        '--data', 'data/processed/flir_thermal_person_fair/data_fair.yaml',
        '--model', 'rtdetr-l.yaml',
        '--epochs', '100',
        '--imgsz', '640',
        '--project', 'results/flir_person_rtdetr-l',
        '--name', 'exp',
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
