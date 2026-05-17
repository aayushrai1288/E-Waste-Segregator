#!/bin/bash

# Training the custom YOLO classification model
cd yolov9_custom

# Using the dataset moved to data/raw/e-waste-yolo-classification
# We need to provide the path relative to the yolov9_custom directory
DATA_PATH="../data/raw/e-waste-yolo-classification"

echo "Starting YOLO Classification Training..."
python classify/train.py --model yolov9-c.pt --data "$DATA_PATH" --epochs 50 --img 224

echo "Training complete. Check runs/train-cls for results."
