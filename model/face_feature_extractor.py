# model/face_feature_extractor.py
import torch
import torch.nn as nn
import torchvision.models as models

import hyperparameters as hp

def build_face_feature_extractor(model_name=hp.FACE_EXTRACTOR_MODEL, pretrained=True, freeze=hp.FREEZE_FACE_EXTRACTOR):
    """Builds a pre-trained ResNet model for feature extraction."""
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    else:
        raise ValueError(f"Unsupported ResNet model: {model_name}")

    # Remove the final fully connected layer (classification layer)
    model.fc = nn.Identity()

    # Freeze layers if required
    if freeze and pretrained:
        print(f"Freezing layers of pre-trained {model_name}")
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the last layers if needed for fine-tuning (optional)
        # for param in model.layer4.parameters(): # Example for ResNet
        #     param.requires_grad = True

    return model