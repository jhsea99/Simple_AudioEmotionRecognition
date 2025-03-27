# face_feature_extractor.py

import torch
import torch.nn as nn
from torchvision import models
from hyperparameters import RESNET_DEPTH, RESNET_PRETRAINED

class FaceFeatureExtractor(nn.Module):
    def __init__(self):
        super(FaceFeatureExtractor, self).__init__()
        if RESNET_DEPTH == 18:
            self.resnet = models.resnet18(pretrained=RESNET_PRETRAINED)
        elif RESNET_DEPTH == 34:
            self.resnet = models.resnet34(pretrained=RESNET_PRETRAINED)
        elif RESNET_DEPTH == 50:
            self.resnet = models.resnet50(pretrained=RESNET_PRETRAINED)
        elif RESNET_DEPTH == 101:
            self.resnet = models.resnet101(pretrained=RESNET_PRETRAINED)
        elif RESNET_DEPTH == 152:
            self.resnet = models.resnet152(pretrained=RESNET_PRETRAINED)
        else:
            raise ValueError(f"Unsupported ResNet depth: {RESNET_DEPTH}. Choose from [18, 34, 50, 101, 152]")

        # Remove the last fully connected layer of ResNet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        # x should have shape (batch_size, 3, H, W), where H and W are defined in ALIGNED_FACE_SIZE
        features = self.resnet(x)
        # The output will have shape (batch_size, C, 1, 1), where C is the number of output channels
        features = torch.flatten(features, 1) # Flatten to (batch_size, C)
        return features

if __name__ == '__main__':
    # Example usage
    import numpy as np
    from hyperparameters import ALIGNED_FACE_SIZE
    dummy_input = torch.randn(2, 3, ALIGNED_FACE_SIZE[0], ALIGNED_FACE_SIZE[1])
    model = FaceFeatureExtractor()
    output = model(dummy_input)
    print("ResNet output shape:", output.shape)
