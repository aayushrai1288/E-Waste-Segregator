#!/bin/bash

# Clone YOLOv9 repository if not already present
if [ ! -d "yolov9" ]; then
    git clone https://github.com/WongKinYiu/yolov9.git
fi

cd yolov9
pip install -r requirements.txt

# Download weights
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt

# Train
python train_dual.py --workers 8 --batch 16 --img 256 --epochs 50 \
--data ../data/raw/yolov9_dataset/data.yaml \
--weights yolov9-e.pt --device 0 \
--cfg models/detect/yolov9.yaml \
--hyp data/hyps/hyp.scratch-high.yaml
