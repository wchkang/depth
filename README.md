# Adaptive Depth Networks with Skippable Sub-Paths

This is the official implementation of [Adaptive Depth Networks with Skippable Sub-Paths (NeurIPS 2024)](https://arxiv.org/abs/2312.16392). 

![fig1-small](./figures/fig1-small.png)
* A single network is trained to have multiple skippable layers.
* At test time, the network's depth can be scaled instantly without any additional cost. 
* Sub-networks of different depths outperform counterpart individual networks.

## Model Zoo

Performance on ILSVRC-2012 validation set. 
Download the pretrained model in the link.

| Model                   | Acc@1 | FLOPs  |                                                                                                 |
| ----------------------- | ----- | ------ | ----------------------------------------------------------------------------------------------- |
| ResNet50-ADN (FFFF)     | 77.6% | 4.11G  | [Donwload](https://drive.google.com/file/d/1thbJDkDYhhM7ZI3LY8d9dZ4TWSbYMT0b/view?usp=sharing)  |
| ResNet50-ADN (TTTT)     | 76.1% | 2.58G  |                                                                                                 |
| MobileNetV2-ADN (FFFFF) | 72.5% | 0.32G  | [Donwload](https://drive.google.com/file/d/1bft5SECYXOFjEhPSkAp2Z9d1U-7w2Mnz/view?usp=sharing) |
| MobileNetV2-ADN (TTTTT) | 70.6% | 0.22G  |                                                                                                 |
| ViT-b/16-ADN (FFFF)     | 81.4% | 17.58G | [Download](https://drive.google.com/file/d/1DlHNgjDCKJOWWFSuQIjClA5Ewbc6Jy3u/view?usp=sharing)  |
| ViT-b/16-ADN (TTTT)     | 80.6% | 11.76G |                                                                                                 |
| Swin-t-ADN (FFFF)       | 81.6% | 4.49G  | [Download](https://drive.google.com/file/d/10twk67rVBAoKFKZSkgsXEzx1RABX73kF/view?usp=sharing)  |
| Swin-t-ADN (TTTT)       | 78.0% | 2.34G  |                                                                                                 |

## Training and Evaluation on ImageNet
<details>
<summary>Requirements</summary>
 We conducted experiments under:
 <ul>
    <li>python 3.10</li>
    <li>pytorch 2.0, torchvision 0.15</li>
    <li>Cuda 12</li>
  </ul>
</details>
<details>
<summary>Data Preparation</summary>
Download ImageNet2012 train and val images from https://www.image-net.org.
 
We expect the directory structure to be the following:
```
path/to/imagenet2012/
  train/    # train images
  val/      # val images
```
</details>

<details>
<summary>Training</summary>

To train ResNet50-ADN on ILSVRC2012, run this command:
(Add '--fpn' to include intermediate features for self-distillation)

```
torchrun --nproc_per_node=4 train_adn.py --model resnet50 --batch-size 64 --lr-scheduler multisteplr --lr-multi-steps 60 100 140 --epochs 150 --norm-weight-decay 0 --bias-weight-decay 0 --subpath-temp 1.0 --output-dir <checkpoint directory> --data-path <ILSVRC2012 data path> 
```

To train Mobilenet-V2-ADN, run:

```train
torchrun --nproc_per_node=4 train_adn.py --model mobilenet_v2 --epochs 300 --lr 0.1 --wd 0.00001 --lr-scheduler multisteplr --lr-multi-steps 150 225 285 --batch-size 64 --norm-weight-decay 0 --bias-weight-decay 0 --subpath-temp 1.0 --output-dir <checkpoint directory> --data-path <ILSVRC2012 data path>
```

To train Swin-t-ADN, run:

```train
torchrun --nproc_per_node=4 train_adn.py --model swin_t --epochs 300 --batch-size 256 --opt adamw --lr 0.001 --weight-decay 0.05 --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs 20 --lr-warmup-decay 0.01 --amp --label-smoothing 0.1 --mixup-alpha 0.8 --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 --interpolation bicubic --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 224 --subpath-temp 1.0 --output-dir <checkpoint directory> --data-path <ILSVRC2012 data path>
```

To train Vit-b-16-ADN, run:

```train
torchrun --nproc_per_node=4 train_adn.py --model vit_b_16 --epochs 300 --batch-size 256 --opt adamw --lr 0.00075 --wd 0.2 --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30 --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema --subpath-temp 1.0 --output-dir <checkpoint directory> --data-path <ILSVRC2012 data path>
```
</details>

<details>
<summary>Evaluation</summary>

change *--skip-cfg* to select different sub-networks.

For example, use *'--skip-cfg True True True True'* to select the smallest sub-network.

To evaluate ResNet50-ADN, run:

```eval
python train_adn.py --model resnet50 --test-only --weights <weights file> --batch-size 256 --skip-cfg False False False False  --data-path <ILSVRC-2012 data path>
```

To evaluate MobileNetV2-ADN, run:

```eval
python train_adn.py --model mobilenet_v2 --test-only --weights <weights file> --batch-size 256 --skip-cfg False False False False False --data-path <ILSVRC-2012 data path>
```

To evaluate Swin-T-ADN, run:

```eval
python train_adn.py --model swin_t --test-only --weights <weights file> --batch-size 256 --skip-cfg False False False False --model-ema --interpolation bicubic --data-path <ILSVRC-2012 data path>
```

To evaluate Vit-b-16-ADN, run:

```eval
python train_adn.py --model vit_b_16 --test-only --weights <weights file> --batch-size 256 --skip-cfg False False False False --model-ema --data-path <ILSVRC-2012 data path>
```

</details>

## Citation

Please use the following BibTeX entries:

```
@inproceedings{kang2024adaptivedepth,
  title={Adaptive Depth Networks with Skippable Sub-Paths},
  author={Kang, Woochul and Lee, Hyungseop},
  booktitle={International Conference on Neural Information Processing (NeurIPS)},
  year={2024},
}
```
