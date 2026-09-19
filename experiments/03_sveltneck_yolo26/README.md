# Experiment 03 — SvelteNeck-YOLO26

Purpose: Core research contribution.

Architecture:
YOLO26n backbone + SvelteNeck + native YOLO26 detection head.

SvelteNeck:
P3: 128x80x80 -> 64x80x80
P4: 128x40x40 -> 128x40x40
P5: 256x20x20 -> 256x20x20

Current expansion ratio: e=0.50

Protocol:
- Image size: 640
- Batch: 8
- Epochs: 30
- Seed: 42
- Deterministic: True
- AMP: True
- GPU: Tesla T4
- Framework: Ultralytics

Never initialize this experiment from a previous SvelteNeck checkpoint.
