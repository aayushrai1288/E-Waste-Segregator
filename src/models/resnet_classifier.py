import torch
import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes=10, pretrained=True):
    # Load pre-trained ResNet-50
    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50()
    
    # Modify the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == "__main__":
    model = get_resnet50(num_classes=10)
    print(model)
