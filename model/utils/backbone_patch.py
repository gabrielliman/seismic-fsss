import torch
import torch.nn as nn

def convert_resnet_to_grayscale(backbone):
    """
    Adapts the first conv layer (conv1) of a torchvision ResNet to accept
    1 input channel instead of 3 (seismic amplitude patches are grayscale).
    """
    old_conv = backbone.conv1
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    # initialise the weights as the mean over the RGB channels
    with torch.no_grad():
        new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias[:] = old_conv.bias

    backbone.conv1 = new_conv
    return backbone

