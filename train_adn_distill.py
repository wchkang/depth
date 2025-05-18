# Adapted from Pytorch training script for ImageNet classification
# by Woochul Kang (wchkang@inu.ac.kr)

import datetime
import os
import time
import warnings

import presets
import torch
import torch.utils.data
import torchvision
import transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

from models._utils_fpn import IntermediateLayerGetter
from models.misc import FrozenBatchNorm2d
import timm # for ResNext101

import torch.nn.functional as F
import sys

import models

from utils_dist import get_dist_info, is_master, print_at_master
from semantic.metrics import AccuracySemanticSoftmaxMet
from semantic.semantic_loss import SemanticSoftmaxLoss, SemanticKDLoss
from semantic.semantics import ImageNet21kSemanticSoftmax

from utils_criterion import JSD

def train_one_epoch_twobackward(
    model, 
    criterion, 
    criterion_kd, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    args, 
    model_ema=None, 
    scaler=None,
    skip_cfg_basenet=None,
    skip_cfg_supernet=None,
    subpath_alpha=0.5,
    subpath_temp=1.0,
    fpn=False,
    ):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        alpha = subpath_alpha

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # forward pass for super_net
            outputs_full = model(image, skip=skip_cfg_supernet)  # original

            if fpn:
                outputs_full_features = outputs_full["features"]
                outputs_full = outputs_full["model_out"]
            
            outputs_full_topK, pred_full = outputs_full.topk(500, 1, largest=True, sorted=True)
            
            loss_full= criterion(outputs_full, target)

            loss_full = alpha * loss_full
            
            with torch.cuda.amp.autocast(enabled=False):
                if scaler is not None:
                    scaler.scale(loss_full).backward()
                else:
                    loss_full.backward()
            
            # forward pass for base_net
            outputs_skip = model(image, skip=skip_cfg_basenet)

            if fpn:
                outputs_skip_features = outputs_skip["features"]
                outputs_skip = outputs_skip["model_out"]

            T = subpath_temp

            loss_feature_kd = 0

            # get feature KD loss. Only ResNet50 is supported.
            if fpn:
                for k, _ in outputs_full_features.items():
                    loss_feature_kd += criterion_kd(F.log_softmax(outputs_skip_features[k]/T, dim=1), F.softmax(outputs_full_features[k].clone().detach()/T, dim=1)) * T*T

            # get softmax KD loss
            outputs_skip_topK = outputs_skip.gather(1, pred_full)
            # this is used for res50 experiments 
            loss_softmax_kd = criterion_kd(F.log_softmax(outputs_skip_topK[:,0:500]/T, dim=1), F.softmax(outputs_full_topK[:,0:500].clone().detach()/T, dim=1)) * T*T
                    
            # final loss
            loss_kd = (1. - alpha) * (loss_softmax_kd  + loss_feature_kd)

        if scaler is not None:
            scaler.scale(loss_kd).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_kd.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(outputs_full, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["loss_full_acc"].update(loss_full.item(), n=batch_size)
        metric_logger.meters["loss_softmax_kd"].update(loss_kd.item(), n=batch_size)
        if fpn:
            metric_logger.meters["loss_features_kd"].update(loss_feature_kd.item(), n=batch_size)
        # metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        acc1_skip, acc5_skip = utils.accuracy(outputs_skip, target, topk=(1, 5))
        # metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1_skip"].update(acc1_skip.item(), n=batch_size)
        metric_logger.meters["acc5_skip"].update(acc5_skip.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        sys.stdout.flush()

def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix="", skip=None, fpn=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"[subnet]{skip} Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image, skip=skip)
            if fpn:
                output = output["model_out"]
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


# original
def train_one_epoch_twobackward_external_teacher(
    model_teacher,
    model, 
    criterion,
    criterion_kd, 
    criterion_jsd, # experiment
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    args, 
    model_ema=None, 
    scaler=None,
    skip_cfg_basenet=None,
    skip_cfg_supernet=None,
    subpath_alpha=0.5,
    subpath_temp_teacher_full=1.0,
    subpath_temp_full_base=1.0
    ):
    model_teacher.eval()
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        alpha = subpath_alpha

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            top_k = 500 # default: 500
            # forward pass for the teacher
            with torch.no_grad():
               outputs_teacher = model_teacher(image) 
            outputs_teacher_topK, pred_teacher = outputs_teacher.topk(top_k, 1, largest=True, sorted=True)
            
            # forward pass for super_net
            outputs_full = model(image, skip=skip_cfg_supernet)  # original         
            outputs_full_topK, pred_full = outputs_full.topk(top_k, 1, largest=True, sorted=True)

            # loss_full= criterion(outputs_full, target)
            
            T = subpath_temp_teacher_full * 2.0 # experiment: 2025.04.24
         
            # get softmax KD loss between the teacher and the super
            outputs_full_topK = outputs_full.gather(1, pred_teacher) 
            # orig KD
            loss_softmax_kd_teacher_full = criterion_kd(F.log_softmax(outputs_full_topK[:,0:top_k]/T, dim=1), F.softmax(outputs_teacher_topK[:,0:top_k].clone().detach()/T, dim=1)) * T*T
            # experiment JSD #1 => seems not working
            # loss_softmax_kd_teacher_full = criterion_jsd(outputs_full_topK[:,0:top_k], outputs_teacher_topK[:,0:top_k].clone().detach())
            
            # original
            # loss_full = alpha * loss_softmax_kd_teacher_full 

            # exp: mix ce and kd
            # kd_ce_alpha = 0.9 # step 1 pretraining imagenet1k
            kd_ce_alpha = 0.7 # step 3 finetuning imagenet1k
            loss_ce_full = criterion(outputs_full, target)
            loss_full = alpha * (kd_ce_alpha * loss_softmax_kd_teacher_full + (1 - kd_ce_alpha) * loss_ce_full)
            
            with torch.cuda.amp.autocast(enabled=False):
                if scaler is not None:
                    scaler.scale(loss_full).backward()
                else:
                    loss_full.backward()
            
            # forward pass for base_net
            outputs_skip = model(image, skip=skip_cfg_basenet)

            T = subpath_temp_full_base  * 2.0 # experiment: 2025.03.26

            # orig #1: get softmax KD loss between the super and the base
            outputs_skip_topK = outputs_skip.gather(1, pred_full)
            loss_softmax_kd_full_base = criterion_kd(F.log_softmax(outputs_skip_topK[:,0:top_k]/T, dim=1), F.softmax(outputs_full_topK[:,0:top_k].clone().detach()/T, dim=1)) * T*T

            # exp: get softmax KD loss between the teacher and the base
            # outputs_skip_topK = outputs_skip.gather(1, pred_teacher)
            # loss_softmax_kd_full_base = criterion_kd(F.log_softmax(outputs_skip_topK[:,0:500]/T, dim=1), F.softmax(outputs_teacher_topK[:,0:500].clone().detach()/T, dim=1)) * T*T

            # final loss
            loss_skip = (1 - alpha) * loss_softmax_kd_full_base

        if scaler is not None:
            scaler.scale(loss_skip).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_skip.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        # acc1, acc5 = utils.accuracy(outputs_full, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["loss_full"].update(loss_full.item(), n=batch_size)
        metric_logger.meters["loss_skip"].update(loss_skip.item(), n=batch_size)
        metric_logger.meters["loss_kd_teacher_full"].update(loss_softmax_kd_teacher_full.item(), n=batch_size)
        metric_logger.meters["loss_ce_full"].update(loss_ce_full.item(), n=batch_size)
        metric_logger.meters["loss_kd_full_base"].update(loss_softmax_kd_full_base.item(), n=batch_size)
      
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        sys.stdout.flush()

# JSD 
def train_one_epoch_twobackward_external_teacher_JSD(
    model_teacher,
    model, 
    criterion,
    criterion_kd, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    args, 
    model_ema=None, 
    scaler=None,
    skip_cfg_basenet=None,
    skip_cfg_supernet=None,
    subpath_alpha=0.5,
    subpath_temp_teacher_full=1.0,
    subpath_temp_full_base=1.0
    ):
    model_teacher.eval()
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        alpha = subpath_alpha

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            
            # forward pass for the teacher
            with torch.no_grad():
               outputs_teacher = model_teacher(image) 
            outputs_teacher_topK, pred_teacher = outputs_teacher.topk(500, 1, largest=True, sorted=True)
            
            # forward pass for super_net
            outputs_full = model(image, skip=skip_cfg_supernet)  # original         
            outputs_full_topK, pred_full = outputs_full.topk(500, 1, largest=True, sorted=True)

            # loss_full= criterion(outputs_full, target)
            
            T = subpath_temp_teacher_full
         
            # get softmax KD loss between the teacher and the super
            outputs_full_topK = outputs_full.gather(1, pred_teacher) 
            # loss_softmax_kd_teacher_full = criterion_kd(F.log_softmax(outputs_full_topK[:,0:500]/T, dim=1), F.softmax(outputs_teacher_topK[:,0:500].clone().detach()/T, dim=1)) * T*T

            loss_softmax_kd_teacher_full = criterion_kd(outputs_full_topK[:,0:500], outputs_teacher_topK[:,0:500].clone().detach()) 


            loss_full = alpha * loss_softmax_kd_teacher_full
            
            with torch.cuda.amp.autocast(enabled=False):
                if scaler is not None:
                    scaler.scale(loss_full).backward()
                else:
                    loss_full.backward()
            
            # forward pass for base_net
            outputs_skip = model(image, skip=skip_cfg_basenet)

            T = subpath_temp_full_base

            # orig #1: get softmax KD loss between the super and the base
            outputs_skip_topK = outputs_skip.gather(1, pred_full)

            # JSD           
            loss_softmax_kd_full_base = criterion_kd(outputs_skip_topK[:,0:500], outputs_full_topK[:,0:500].clone().detach())

            # exp: get softmax KD loss between the teacher and the base
            # outputs_skip_topK = outputs_skip.gather(1, pred_teacher)
            # loss_softmax_kd_full_base = criterion_kd(F.log_softmax(outputs_skip_topK[:,0:500]/T, dim=1), F.softmax(outputs_teacher_topK[:,0:500].clone().detach()/T, dim=1)) * T*T

            # final loss
            loss_skip = (1 - alpha) * loss_softmax_kd_full_base

        if scaler is not None:
            scaler.scale(loss_skip).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_skip.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        # acc1, acc5 = utils.accuracy(outputs_full, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["loss_full"].update(loss_full.item(), n=batch_size)
        metric_logger.meters["loss_skip"].update(loss_skip.item(), n=batch_size)
        metric_logger.meters["loss_kd_teacher_full"].update(loss_softmax_kd_teacher_full.item(), n=batch_size)
        metric_logger.meters["loss_kd_full_base"].update(loss_softmax_kd_full_base.item(), n=batch_size)
      
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        sys.stdout.flush()


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = args.ra_magnitude
        augmix_severity = args.augmix_severity
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
            ),
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path)
    else:
        preprocessing = presets.ClassificationPresetEval(
            crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
        )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def freeze_parameters(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def freeze_norm(m: nn.Module):
    if isinstance(m, nn.BatchNorm2d):
        m = FrozenBatchNorm2d(m.num_features)
    else:
        for name, child in m.named_children():
            _child = freeze_norm(child)
            if _child is not child:
                setattr(m, name, _child)
    return m  


def load_model_weights(model, model_path, freeze_bn=False):
    if freeze_bn:
        print("freeze norm")
        model = freeze_norm(model)
    state = torch.load(model_path, map_location='cpu')
    # print(state.keys())
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            print('skipping num_batches_tracked')
            continue
        p = model.state_dict()[key]
        if key in state.keys():
            ip = state[key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print(
                    'could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
       
    return model

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # exp: distill using imagenet21k and evalute using imagenet1k
    # train_dir = os.path.join("~/data/imagenet21k_resized/", "train_val_small_classes")
    # train_dir = os.path.join("~/data/imagenet21k_resized/", "train_val")
    # train_dir = os.path.join("/media/data/imagenet21k_resized/", "imagenet21k-1k-merged")
    # train_dir = os.path.join("/media/data/", "imagenet21k-1k-merged")
    # train_dir = os.path.join("~/data/imagenet21k_resized/", "imagenet21k-1k-merged")
    train_dir = os.path.join("/media/data/ILSVRC2012/", "train")
    val_dir = os.path.join("/media/data/ILSVRC2012/", "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    # experimetn to use both 1k and 21k alternatively
    # dataset_21k, _, train_21k_sampler, _ = load_data(train_dir_21k, val_dir, args)
    
    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))
        
    # print("dataset test")
    # print(dataset[11000][0].shape)
    # print(dataset[11000][1])

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # experiment to use both 21k and 1k 
    # data_loader_21k = torch.utils.data.DataLoader(
    #     dataset_21k,
    #     batch_size=args.batch_size,
    #     sampler=train_21k_sampler,
    #     num_workers=args.workers,
    #     pin_memory=True,
    #     collate_fn=collate_fn,
    # )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating teacher")
    # weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2
    # model_teacher = torchvision.models.resnet101(weights=weights)
    # weights = torchvision.models.EfficientNet_V2_S_Weights
    # model_teacher = torchvision.models.efficientnet_v2_s(weights=weights)

    # ResNext101
    # weights = torchvision.models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
    # model_teacher = torchvision.models.resnext101_64x4d(weights=weights)
    # print("ResNext101")

    # ConvNext_Large
    # weights = torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
    # model_teacher = torchvision.models.convnext_large(weights=weights)
    # print("ConvNext_Large")

    # ConvNext_Base
    # weights = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
    # model_teacher = torchvision.models.convnext_base(weights=weights)
    # print(ConvNext_Base)

    # ResNext101
    # this is imagenet-1k supervised training
    # model_teacher = timm.create_model('resnext101_32x16d', pretrained=True)
    # print("ResNext101_32x16d")
    # this is semi weakly supervised training. Acc 85.1%
    
    # swsl top1 83.34
    # model_teacher = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_swsl')

    # swsl top1 84.28
    print("ResNext101_32x8d_swsl")
    model_teacher = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_swsl')

    # wsl top1 84.16
    # model_teacher = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')

    # PResNet101
    # checkpoint = torch.load("./pretrained/ResNet101_vd_ssld_pretrained.pth")
    # model_teacher = models.PResNet(depth=101, pretrained=False)
    # model_teacher.load_state_dict(checkpoint)
    # print("PResNet101")

    print("Creating model")
    if args.model not in models.__dict__.keys():
        print(f"{args.model} is not supported")
        sys.exit()
    
    if args.imagenet21k:
        num_classes =  10450 # fall11 11221
    else:
        num_classes = 1000

    model = models.__dict__[args.model](num_classes=num_classes)

    if args.fpn:
        if args.model.startswith("resnet"):
            model = models.resnet50()

            model = IntermediateLayerGetter(model, ['layer1', 'layer2', 'layer3'])
        else:
            print("[experiment] fpn is only supported for resnet models")
            sys.exit()
        
        model_without_ddp = model.model
    else:
        model_without_ddp = model
    
    if args.weights and args.test_only:
        checkpoint = torch.load(args.weights, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint)
    
    # if args.imagenet21k and args.weights:
    if args.weights:
        print("Loading weights: ", args.weights)
        load_model_weights(model_without_ddp, args.weights, freeze_bn=False)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.imagenet21k and args.freeze_params:
        print("Freeze parameters.")
        freeze_parameters(model)
        freeze_norm(model)
        for param in model.fc.parameters():
            param.requires_grad = True

    model_teacher.to(device)
    model.to(device)

    # semantic
    semantic_softmax_processor = ImageNet21kSemanticSoftmax(args)
    semantic_met = AccuracySemanticSoftmaxMet(semantic_softmax_processor)

    if args.imagenet21k:
        criterion = SemanticSoftmaxLoss(semantic_softmax_processor)
        criterion_kd = SemanticKDLoss(semantic_softmax_processor)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        criterion_kd = nn.KLDivLoss(reduction='batchmean')
        criterion_jsd = JSD()
    
    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "multisteplr":
        main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_multi_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.distributed:
        model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[args.gpu], find_unused_parameters=True)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        if args.fpn:
            model_without_ddp = model.module.model
        else:
            model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

     
    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        skip_cfg = args.skip_cfg
        if model_without_ddp.num_skippable_stages != len(skip_cfg):
            print(f"Error: {args.model} has {model_without_ddp.num_skippable_stages} skippable stages!")
            return

        if args.imagenet21k:
            # if model_ema:
            #     evaluate_imagenet21k(model_ema, criterion, data_loader_test, device=device, skip=skip_cfg, met=semantic_met)
            # else:
            #     evaluate_imagenet21k(model, criterion, data_loader_test, device=device, skip=skip_cfg, met=semantic_met)
            return
    
        else:
            if model_ema:
                evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA", skip=skip_cfg, fpn=False)
            else:
                evaluate(model, criterion, data_loader_test, device=device, skip=skip_cfg, fpn=args.fpn)
            return
    
    num_skippable_stages = model_without_ddp.num_skippable_stages

    skip_cfg_basenet = [True for _ in range(num_skippable_stages)]
    skip_cfg_supernet = [False for _ in range(num_skippable_stages)]

    print("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        # train_one_epoch_twobackward(
        #     model, 
        #     criterion, 
        #     criterion_kd, 
        #     optimizer, 
        #     data_loader, 
        #     device, epoch, 
        #     args, 
        #     model_ema, 
        #     scaler, 
        #     skip_cfg_basenet, 
        #     skip_cfg_supernet, 
        #     subpath_alpha=args.subpath_alpha, 
        #     subpath_temp=args.subpath_temp,
        #     fpn=args.fpn
        #     )

        # # experiment: alternate 21k and 1k dataloader
        # if epoch % 2 == 0:
        #     dl = data_loader_21k
        # else:
        #     dl = data_loader

        # # learn from an external teacher
        train_one_epoch_twobackward_external_teacher(
                model_teacher,
                model, 
                criterion,
                criterion_kd, 
                criterion_jsd, # experiment
                optimizer, 
                data_loader,  # experiment 
                device, epoch, 
                args, 
                model_ema, 
                scaler, 
                skip_cfg_basenet, 
                skip_cfg_supernet, 
                subpath_alpha=args.subpath_alpha, 
                subpath_temp_teacher_full=1.0,
                subpath_temp_full_base=1.0,
        )
            
        lr_scheduler.step()

        # if args.imagenet21k:
        #     if model_ema:
        #         evaluate_imagenet21k(model_ema, criterion, data_loader_test, device=device, skip=skip_cfg_basenet, met=semantic_met)
        #         evaluate_imagenet21k(model_ema, criterion, data_loader_test, device=device, skip=skip_cfg_supernet, met=semantic_met)
        #     else:
        #         evaluate_imagenet21k(model, criterion, data_loader_test, device=device, skip=skip_cfg_basenet, met=semantic_met)
        #         evaluate_imagenet21k(model, criterion, data_loader_test, device=device, skip=skip_cfg_supernet, met=semantic_met)    
        # else:
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA", skip=skip_cfg_basenet, fpn=False)
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA", skip=skip_cfg_supernet, fpn=False)
        else:
            evaluate(model, criterion, data_loader_test, device=device, skip=skip_cfg_basenet, fpn=args.fpn)
            evaluate(model, criterion, data_loader_test, device=device, skip=skip_cfg_supernet, fpn=args.fpn)
    
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            if epoch % 10 == 0 or epoch > (args.epochs - 30):
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--subpath-alpha", default=0.5, type=float, help="sub-paths distillation alpha (default: 0.5)")
    parser.add_argument("--subpath-temp", default=1.0, type=float, help="sub-paths distillation temperature (default: 1.0)")
    parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-multi-steps", nargs="+", default=[60,100,140], type=int, help="multi step milestones")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--skip-cfg", nargs="+", default=[False, False, False, False], type=lambda x: x == "True", help="configuration for skip stages")
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # ImageNet21K pretraining
    parser.add_argument("--imagenet21k", action="store_true", help="pretrain with imagenet21k")
    parser.add_argument("--tree_path", default='./resources/imagenet21k_miil_tree.pth', type=str)
    parser.add_argument("--freeze_params", action="store_true", help="freeze parameters and norms except the head")

    # FPN: only support ResNet
    parser.add_argument("--fpn", action="store_true", help="Use FPN in backbone")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
