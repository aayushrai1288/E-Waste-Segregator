import subprocess
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Run YOLO Classification Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to the image')
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLO classification weights (.pt)')
    args = parser.parse_args()

    # Path to the custom yolo directory
    yolo_dir = "yolov9_custom"
    
    # Run the predict.py from yolov9_custom/classify
    cmd = [
        "python", 
        os.path.join(yolo_dir, "classify", "predict.py"),
        "--weights", args.weights,
        "--source", args.image
    ]
    
    print(f"Running inference: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
