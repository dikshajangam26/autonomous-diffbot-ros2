#!/usr/bin/env python3
"""
make_warehouse_model.py - run ONCE to prepare the object detector.

YOLOv8-World is a pre-trained YOLOv8 that can find objects you simply NAME in words. This script
downloads it, tells it which warehouse objects to look for, and saves the result as
/ws/models/warehouse_world.pt. After that the robot never needs the internet (or the
text model CLIP) again.

    python3 /ws/tools/make_warehouse_model.py
    python3 /ws/tools/make_warehouse_model.py --classes "pallet,shelving rack,forklift,person"

(The words you give become the labels. Change them and run it again to look for other things.)
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--classes', default='pallet,shelving rack,cardboard box,person')
parser.add_argument('--base', default='yolov8s-worldv2.pt', help='pre-trained model to start from')
parser.add_argument('--out-dir', default='/ws/models')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
names = [c.strip() for c in args.classes.split(',') if c.strip()]

from ultralytics import YOLOWorld  # noqa: E402  (slow import, so after the argument parsing)

print(f'Loading {args.base} (downloaded the first time, about 50 MB) ...')
model = YOLOWorld(os.path.join(args.out_dir, args.base))
print(f'Setting the words to look for: {names}')
model.set_classes(names)
out = os.path.join(args.out_dir, 'warehouse_world.pt')
model.save(out)
print(f'Saved {out}')
