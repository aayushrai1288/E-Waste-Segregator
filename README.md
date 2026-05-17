# E-Waste Segregator

## Overview
The E-Waste Segregator is a deep learning-based project designed to classify and detect various types of electronic waste. The project leverages state-of-the-art architectures like ResNet-50, VGG16, and YOLOv9 to automate the identification of e-waste items, facilitating better recycling and waste management.

## Project Structure
- `data/`: Contains raw and processed datasets.
- `src/`: Core source code for models, utilities, and training.
  - `models/`: Implementations of ResNet, VGG, and YOLOv9 wrappers.
  - `utils/`: Data preprocessing, loading, and visualization scripts.
- `notebooks/`: Experimental Jupyter notebooks for classification and detection.
- `scripts/`: Shell scripts for training and inference.
- `requirements.txt`: List of dependencies.

## Models
1. **ResNet-50**: Achieved high accuracy (~98%) for classification across 10 classes.
2. **VGG16**: Used for comparative analysis in classification tasks.
3. **YOLOv9**: Employed for real-time object detection and localization of e-waste items.

## Dataset
The dataset consists of approximately 2400 training images and 300-400 testing/validation images, covering 10 classes of electronic waste (e.g., Mouse, Keyboard, Laptop, etc.).
Dataset is hosted on Roboflow: [E-waste Segregator](https://universe.roboflow.com/rajarshi/ds1-znqts).

## Setup and Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset:
   ```bash
   python src/utils/download_dataset.py
   ```

## Usage
### Training
To train the classification model:
```bash
python src/train.py --model resnet50 --epochs 25
```

### Detection
To run YOLOv9 detection:
```bash
bash scripts/train_yolo.sh
```

## Team
- Rita Shukla
- Kanishka Goyal
- Aayush Rai
"# E-Waste-Segregator" 
