# Adapted from Pytorch implementation of ResNet
# by Woochul Kang (wchkang@inu.ac.kr)
# Check Neurips 2024 paper for details.

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from .transforms._presets import ImageClassification
from .utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface


__all__ = [
    "ResNet",
    "resnet50",
    "resnet50vd", # variant 'd' for downsample
    "resnet101",
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        skippable: bool = False, # is this block skippable? @woochul
    ) -> None:
        super().__init__()

        self.skippable = skippable

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if self.skippable == False:   # Switch BN for shared layers.  
            self.bn1_skip = norm_layer(width)
            self.bn2_skip = norm_layer(width)
            self.bn3_skip = norm_layer(planes * self.expansion)
            

    def forward(self, x: Tensor, skip: bool = False) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1_skip(out) if self.skippable == False and skip == True else self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2_skip(out) if self.skippable == False and skip == True else self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3_skip(out) if self.skippable == False and skip == True else self.bn3(out)
            
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SkippableSequentialBlocks(nn.Sequential):
    """Skips some blocks in the stage"""
    def forward(self, input, skip = False):
        """Extends nn.Sequential's forward for skipping some blocks
        Args:
            x (Tensor): input tensor
            skip (bool): if True, skip the last half blocks in the stage.
        """
        for i in range(len(self)):
            if self[i].skippable == True and skip == True:
                pass
            else:
                input = self[i](input, skip)
                
        return input
    

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[Bottleneck,]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        variant: str = 'a', # 'a', or 'd' for downsample
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        self.num_skippable_stages = len(layers)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.variant = variant # 'a', or 'd' for downsample

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten() # for fpn, use nn operator instead of torch.flatten
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[Bottleneck,]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride ==2 and self.variant == 'd':
                downsample = nn.Sequential(
                    nn.AvgPool2d(2, 2, 0, ceil_mode=True),
                    conv1x1(self.inplanes, planes * block.expansion, stride=1),
                    norm_layer(planes * block.expansion),
                )
                print(f"Downsample: {self.variant}")
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []

        # default: n_shared == n_skippable
        n_shared = (blocks + 1) // 2  
        n_shared = min(blocks - 1, n_shared) # maximum is blocks - 1

        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, skippable=False
            )
        )
        self.inplanes = planes * block.expansion
        for i_block in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    skippable = (i_block >= n_shared),
                )
            )

        return SkippableSequentialBlocks(*layers)  
    

    def _forward_impl(self, x: Tensor, skip: List[bool] = None) -> Tensor:
        # See note [TorchScript super()]

        assert self.num_skippable_stages == len(skip), \
            f"The networks has {self.num_stages} skippable stages, got: {len(skip)}"

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, skip[0])
        x = self.layer2(x, skip[1])
        x = self.layer3(x, skip[2])
        x = self.layer4(x, skip[3])

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor, skip: List[bool] = None) -> Tensor:
        if skip is None:
            skip = [False for _ in range(self.num_skippable_stages)]
        return self._forward_impl(x, skip = skip)


def _resnet(
    block: Type[Union[Bottleneck,]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model

def resnet50(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)

def resnet50vd(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs['variant'] = 'd'
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def resnet101(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


