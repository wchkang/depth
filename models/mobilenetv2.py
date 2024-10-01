# Adapted from Pytorch implementation of MobileNetV2
# by Woochul Kang (wchkang@inu.ac.kr)
# Check Neurips 2024 paper for details.

from functools import partial
from typing import Any, Callable, List, Optional

import torch
from torch import nn, Tensor

from .misc import SkippableConv2dNormActivation, Conv2dNormActivation
from .transforms._presets import ImageClassification
from .utils import _log_api_usage_once
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible


__all__ = ["MobileNetV2", "mobilenet_v2"]


# necessary for backwards compatibility
class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None,
        skippable: bool = False
    ) -> None:
        super().__init__()
        
        self.skippable = skippable # @woochul

        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                SkippableConv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6, skippable=skippable)
            )
        layers.extend(
            [
                # dw
                SkippableConv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                    skippable=skippable
                ),
                # pw-linear
                # nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # norm_layer(oup),
                SkippableConv2dNormActivation(
                    in_channels=hidden_dim,
                    out_channels=oup,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_layer=norm_layer,
                    activation_layer=None,
                    skippable=skippable
                ),
            ]
        )
        self.conv = SkippableSequentialLayers(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor, skip: bool = False) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x, skip)
        else:
            return self.conv(x, skip)


class SkippableSequentialLayers(nn.Sequential):
    """Pass 'skip' flag to the layer with switchable BNs"""
    def forward(self, input, skip = False):
        for i in range(len(self)):
            if isinstance(self[i], SkippableConv2dNormActivation):
                input = self[i](input, skip)
            else:
                input = self[i](input)
        return input

class SkippableSequentialBlocks(nn.Sequential):
    """Skips some blocks in the stage"""
    def forward(self, input, skip = False):
        for i in range(len(self)):
            if self[i].skippable == True and skip == True:
                pass
            else:
                input = self[i](input, skip)
        return input

class SkippableSequentialStages(nn.Sequential):
    """Pass 'skip' flag to selected stages"""
    def __init__(self, num_skippable_stages, *args):
        super().__init__(*args)
        self.num_skippable_stages = num_skippable_stages

    def forward(self, input, skip: List[bool] = None):
        # The first 2 stages and the last 2 stages are not skippable in MV2.
        # Hence, only accepts 'skip' flags for the skippable stages.
        assert len(skip) == self.num_skippable_stages, \
            f"The networks has {self.num_skippable_stages} skippable stages, got: {len(skip)}"

        input = self[0](input, skip[0]) # 1st stage: non-skippable, switchable BNs 
        input = self[1](input, skip[0]) # 2nd stage: non-skippable, switchable BNs 

        for i in range(2, len(self) - 2):
            input = self[i](input, skip[i-2])
        
        input = self[-2](input, skip[-1]) # penultimate stage: non skippable, switchable BN
        input = self[-1](input, skip[-1]) # last stage: non-skippable, switchable BNs
        
        return input


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability

        """
        super().__init__()
        _log_api_usage_once(self)

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],  
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1], 
            ]

        # Number of skippable stages.
        # First, last stages are not skippable sice they have only 1 block.
        self.num_skippable_stages = len(inverted_residual_setting) - 2 

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            SkippableConv2dNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6, skippable=False)
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)

            # A skippable stage needs at leate 2 blocks. @woochul
            if n >= 2:
                skippable_stage: List[nn.Module] = []
            
            for i in range(n):
                stride = s if i == 0 else 1
                if n >=2:
                    skippable = True if (i >= (n + 1) // 2) else False  # default
                    skippable_stage.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, skippable=skippable))
                else:
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, skippable=False))
                input_channel = output_channel
            
            if n >= 2:
                features.append(SkippableSequentialBlocks(*skippable_stage))
        
        # building last several layers
        features.append(
            SkippableConv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6, skippable=False
            )
        )
        # make it nn.Sequential
        self.features = SkippableSequentialStages(self.num_skippable_stages, *features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor, skip: List[bool] = None) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x, skip)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor, skip: List[bool] = None) -> Tensor:
        if skip == None:
            skip = [True for _ in range(self.num_skippable_stages)]
        return self._forward_impl(x, skip)


def mobilenet_v2(
    *, weights = None, progress: bool = True, **kwargs: Any
) -> MobileNetV2:

    model = MobileNetV2(**kwargs)

    return model
