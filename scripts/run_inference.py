import torch
from torchvision import transforms
from PIL import Image
import os
from models.resnet_classifier import get_resnet50

def predict(image_path, model_path, class_names):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = get_resnet50(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    
    return class_names[top_catid], top_prob.item()

if __name__ == "__main__":
    # Example usage
    # class_names = ['Mouse', 'Keyboard', 'Laptop', ...] 
    # print(predict('test.jpg', 'best_model.pt', class_names))
    pass
