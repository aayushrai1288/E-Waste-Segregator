import torch
from torchvision import transforms
from PIL import Image
import os
import argparse
import sys
from models.resnet_classifier import get_resnet50

def get_prediction(image_path, model_path, classes):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    num_classes = len(classes)
    model = get_resnet50(num_classes=num_classes)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
    model.to(device)
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
        
    image = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0][predicted].item()

    return classes[predicted[0]], confidence

def main():
    parser = argparse.ArgumentParser(description='E-waste Classification Tool')
    parser.add_argument('--image', type=str, required=True, help='Path to the image to classify')
    parser.add_argument('--model', type=str, default='models/best_resnet50.pt', help='Path to the trained model weights')
    args = parser.parse_args()

    # Define classes (Ensure these match the training classes)
    classes = [
        'Air-Conditioner', 'Camera', 'Desktop', 'Dishwasher', 'Drone',
        'Headphone', 'Home-Entertainment', 'Keyboard - Mouse', 'Kitchen-Appliance',
        'Laptops', 'Monitors', 'Outdoor-Cooking', 'Oven', 'Perfume',
        'Personal-Care', 'Printer', 'Refrigerator', 'Remote', 'Speaker',
        'Television', 'Vacuum-Cleaner', 'Washing-Machine', 'Watch', 'Webcam'
    ]

    print(f"\nAnalyzing image: {args.image}...")
    result = get_prediction(args.image, args.model, classes)
    
    if result:
        class_name, conf = result
        print("="*30)
        print(f"Result: {class_name}")
        print(f"Confidence: {conf*100:.2f}%")
        print("="*30)
    else:
        print("Failed to get prediction.")

if __name__ == "__main__":
    # Add project src to path so imports work
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    main()
