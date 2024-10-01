from collections import OrderedDict

import torch
from torch import nn
from typing import Dict, List, Tuple

__all__ = [
    "IntermediateLayerGetter",
]

class IntermediateLayerGetter(nn.Module):
    """
    Module wrapper that returns intermediate layers from a model
    """

    def __init__(self, model: nn.Module, return_layers: List[str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        super(IntermediateLayerGetter, self).__init__()

        self.model = model
        self.return_layers = return_layers


    def forward(self, x, skip=(False, False, False, False)):
        intermedia_features = OrderedDict()
        #print(self.items())
        out = {}
        num_layer = 0
        for name, module in self.model.named_children():
            #if ('layer' in name):
            if ('_skippable' in name):
                # print("skip:", name)
                # print(f"num_layer: {num_layer}")
                x = module(x, skip=skip[num_layer])
                # print(f"skip: {name}: skip={skip[num_layer]}, size:{x.size()}")
                num_layer += 1
            else:
                x = module(x)
                # print(f"noskip: {name}, size:{x.size()}")
            if name in self.return_layers:
                intermedia_features[name] = torch.squeeze(torch.nn.functional.adaptive_avg_pool2d(x,(1,1)))
        out['model_out'] = x
        out['features'] = intermedia_features

        return out

