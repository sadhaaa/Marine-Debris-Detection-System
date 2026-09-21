"""
Batch apply underwater degradation compensation (Beer-Lambert + CLAHE)
to create the enhanced TrashCan dataset split.
"""

import os
import sys
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.underwater_augment import underwater_degradation_compensation

SRC_DIR = ROOT / "data" / "processed" / "trashcan_instance"
DST_DIR = ROOT / "data" / "processed" / "trashcan_instance_uw_enhanced"



def process_image(src_img_path, dst_img_path):
    if dst_img_path.exists():
        return
    img = cv2.imread(str(src_img_path))
    if img is None:
        return
    enhanced = underwater_degradation_compensation(img)
    dst_img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_img_path), enhanced)


def main():
    print(f"Enhancing dataset from {SRC_DIR} to {DST_DIR}...")
    for split in ["val"]:  # Start with val for instant evaluation / testing
        src_imgs = list((SRC_DIR / "images" / split).glob("*.jpg"))
        print(f"Processing {len(src_imgs)} {split} images...")
        
        # Ensure labels are mirrored / symlinked
        dst_lbl_dir = DST_DIR / "labels" / split
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        src_lbl_dir = SRC_DIR / "labels" / split
        for lbl in src_lbl_dir.glob("*.txt"):
            dst_lbl = dst_lbl_dir / lbl.name
            if not dst_lbl.exists():
                os.link(str(lbl), str(dst_lbl)) if hasattr(os, 'link') else dst_lbl.write_bytes(lbl.read_bytes())
                
        dst_img_dir = DST_DIR / "images" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [
                ex.submit(process_image, p, dst_img_dir / p.name)
                for p in src_imgs
            ]
            for f in futures:
                f.result()
        print(f"Done {split} split.")


if __name__ == "__main__":
    main()
