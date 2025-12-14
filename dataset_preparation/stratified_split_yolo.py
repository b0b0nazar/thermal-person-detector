"""
Script to create a fair, stratified 80/20 train/val split for YOLO dataset.
Splits both positive (with person) and negative (no person) images proportionally.
Copies images and labels to new folders: images/train, images/val, labels/train, labels/val.
"""
import shutil
from pathlib import Path
import random

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)

def get_label_type(label_path):
    return 'pos' if label_path.stat().st_size > 0 else 'neg'

def stratified_split(label_dir, split_ratio=0.8):
    pos = [f for f in label_dir.glob('*.txt') if f.stat().st_size > 0]
    neg = [f for f in label_dir.glob('*.txt') if f.stat().st_size == 0]
    random.shuffle(pos)
    random.shuffle(neg)
    n_pos_train = int(len(pos) * split_ratio)
    n_neg_train = int(len(neg) * split_ratio)
    train = pos[:n_pos_train] + neg[:n_neg_train]
    val = pos[n_pos_train:] + neg[n_neg_train:]
    random.shuffle(train)
    random.shuffle(val)
    return train, val

def copy_split_files(split, label_files, raw_img_dir, out_img_dir, out_lbl_dir):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    for lbl_path in label_files:
        img_name = lbl_path.stem + '.jpeg'  # assumes .jpeg extension
        src_img = raw_img_dir / img_name
        dst_img = out_img_dir / img_name
        dst_lbl = out_lbl_dir / lbl_path.name
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
            shutil.copy2(lbl_path, dst_lbl)
        else:
            print(f"Warning: Image not found for label {lbl_path.name}")

def main(raw_img_dir=Path('data/processed/flir_thermal_person/images/train'),
         raw_lbl_dir=Path('data/processed/flir_thermal_person/labels/train'),
         out_root=Path('data/processed/flir_thermal_person_fair'),
         split_ratio=0.8):
    all_lbl_dir = raw_lbl_dir.parent
    all_img_dir = raw_img_dir.parent
    # Combine all labels/images from train and val
    all_lbls = list((all_lbl_dir / 'train').glob('*.txt')) + list((all_lbl_dir / 'val').glob('*.txt'))
    all_imgs = list((all_img_dir / 'train').glob('*.jpeg')) + list((all_img_dir / 'val').glob('*.jpeg'))
    # Create a temp dir to hold all labels
    temp_lbl_dir = out_root / 'all_labels'
    temp_img_dir = out_root / 'all_images'
    temp_lbl_dir.mkdir(parents=True, exist_ok=True)
    temp_img_dir.mkdir(parents=True, exist_ok=True)
    for lbl in all_lbls:
        shutil.copy2(lbl, temp_lbl_dir / lbl.name)
    for img in all_imgs:
        shutil.copy2(img, temp_img_dir / img.name)
    # Stratified split
    train_lbls, val_lbls = stratified_split(temp_lbl_dir, split_ratio)
    # Copy to new split folders
    copy_split_files('train', train_lbls, temp_img_dir, out_root / 'images/train', out_root / 'labels/train')
    copy_split_files('val', val_lbls, temp_img_dir, out_root / 'images/val', out_root / 'labels/val')
    print(f"Done. Train: {len(train_lbls)} images, Val: {len(val_lbls)} images.")

if __name__ == "__main__":
    main()
