import torch
from torch.nn import Linear

try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model

@register_model
def resnet18(pretrained, num_classes, *args, **kwargs):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
    model.fc = Linear(512, num_classes)
    return model

@register_model
def resnet50(pretrained, num_classes, *args, **kwargs):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
    model.fc = Linear(2048, num_classes)
    return model


