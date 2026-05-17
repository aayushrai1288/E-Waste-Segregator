import os
from roboflow import Roboflow

def download_dataset():
    # Use the API key from the project context
    api_key = "jPIIRd35QhC7qco6EgXT"
    rf = Roboflow(api_key=api_key)
    
    workspace = "rajarshi"
    project_name = "ds1-znqts"
    version_num = 1
    
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    
    # Download as YOLOv9 format for detection
    print("Downloading YOLOv9 dataset...")
    dataset = version.download("yolov9", location="data/raw/yolov9_dataset")
    
    # Note: For classification, we might want a different format or just use the same images.
    print(f"Dataset downloaded to {dataset.location}")

if __name__ == "__main__":
    download_dataset()
