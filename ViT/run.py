# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2020 paper
# Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion
# Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek
# Hoiem, Niraj K. Jha, and Jan Kautz
# --------------------------------------------------------

from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse
import torch
from torch import distributed, nn
import random
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

from torchvision import datasets, transforms
import torchvision

import numpy as np
import torch.cuda.amp as amp
import os
import torchvision.models as models

random.seed(0)


def validate(input, target, model, use_fp16):
    """Perform validation on the validation set"""

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        if use_fp16:
            input = input.half()
            dtype = torch.float16
        else:
            dtype = torch.float32
        if use_fp16:
            with torch.autocast(device_type=model.device, dtype=dtype):
                output = model(input)
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        else:
            output = model(input)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


# Command to run the script:
# python run.py --bs=32 --do_flip --r_feature_scale=0.01 --kl_loss_scale=0.1 --setting_id=0 --lr 0.25
# (OPTIONAL) --l2 0.00001 --tv_l1 0.0 --tv_l2 0.0001 --main_loss_multiplier 1.0 --kl_loss_scale 0.1 --first_ln_multiplier 10.0 --output_dir="vit_inversion_output" --store_best_images


def run(args):
    local_rank = args.local_rank
    no_cuda = args.no_cuda
    use_fp16 = args.fp16

    bs = args.bs
    setting_id = args.setting_id
    jitter = args.jitter

    store_dir = args.store_dir
    store_dir = "./generations/{}".format(store_dir)

    parameters = dict()
    parameters["resolution"] = 224
    parameters["start_noise"] = True
    parameters["random_label"] = args.random_label
    parameters["do_flip"] = args.do_flip
    parameters["store_best_images"] = args.store_best_images

    coefficients = dict()
    coefficients["lr"] = args.lr
    coefficients["r_feature_scale"] = args.r_feature_scale
    coefficients["first_ln_multiplier"] = args.first_ln_multiplier
    coefficients["tv_l1_scale"] = args.tv_l1_scale
    coefficients["tv_l2_scale"] = args.tv_l2_scale
    coefficients["l2_scale"] = args.l2_scale
    coefficients["kl_loss_scale"] = args.kl_loss_scale
    coefficients["main_loss_multiplier"] = args.main_loss_multiplier

    torch.manual_seed(local_rank)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    )

    print("Loading model for inversion...")

    ### load models
    # ! Load from torchvision directly
    net = torchvision.models.vit_b_16(weights="IMAGENET1K_V1").to(device)

    if use_fp16:
        net.half()

    for module in net.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.float()

    net.eval()

    # reserved to compute test accuracy on generated images by different networks
    net_verifier = None
    if local_rank == 0:
        print("Loading verifier...")

        # Load ResNet-50
        net_verifier = torchvision.models.resnet50(weights="IMAGENET1K_V1").to(device)

        if use_fp16:
            # Convert model to FP16
            net_verifier.half()

            # Ensure BatchNorm stays in FP32 for stability
            for module in net_verifier.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.float()

        # Set model to evaluation mode
        net_verifier.eval()

    from deepinversion_vit import ViTDeepInversionClass

    criterion = nn.CrossEntropyLoss()

    ViTDeepInversionEngine = ViTDeepInversionClass(
        net_teacher=net,
        net_verifier=net_verifier,
        path=store_dir,
        parameters=parameters,
        setting_id=setting_id,
        bs=bs,
        use_fp16=use_fp16,
        jitter=jitter,
        criterion=criterion,
        coefficients=coefficients,
        network_output_function=lambda x: x,
        hook_for_display=lambda x, y: validate(x, y, net_verifier, use_fp16),
    )
    ViTDeepInversionEngine.generate_batch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        "--rank",
        type=int,
        default=0,
        help="Rank of the current process.",
    )
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--setting_id",
        default=0,
        type=int,
        help="settings for optimization: 0 - multi resolution, 1 - 2k iterations, 2 - 20k iterations",
    )
    parser.add_argument("--bs", default=64, type=int, help="batch size")
    parser.add_argument("--jitter", default=30, type=int, help="input jitter")
    parser.add_argument("--fp16", action="store_true", help="use FP16 for optimization")
    parser.add_argument(
        "--store_dir",
        type=str,
        default="vit",
        help="where to store experimental data",
    )
    parser.add_argument(
        "--do_flip", action="store_true", help="apply flip during model inversion"
    )
    parser.add_argument(
        "--random_label",
        action="store_true",
        help="generate random label for optimization",
    )
    parser.add_argument(
        "--store_best_images",
        action="store_true",
        help="save best images as separate files",
    )

    # Coefficients for optimization
    parser.add_argument(
        "--lr", type=float, default=0.05, help="learning rate for optimization"
    )
    parser.add_argument(
        "--r_feature_scale",
        type=float,
        default=0.05,
        help="coefficient for feature distribution regularization",
    )
    parser.add_argument(
        "--first_ln_multiplier",
        type=float,
        default=10.0,
        help="additional multiplier on first layer of R_feature",
    )
    parser.add_argument(
        "--tv_l1_scale",
        type=float,
        default=0.005,
        help="coefficient for total variation L1 loss",
    )
    parser.add_argument(
        "--tv_l2_scale",
        type=float,
        default=0.05,
        help="coefficient for total variation L2 loss",
    )
    parser.add_argument(
        "--l2_scale", type=float, default=0.001, help="l2 loss on the image"
    )
    parser.add_argument(
        "--kl_loss_scale",
        type=float,
        default=0.15,
        help="Coefficient for KL Loss",
    )
    parser.add_argument(
        "--main_loss_multiplier",
        type=float,
        default=1.0,
        help="coefficient for the main loss in optimization",
    )
    args = parser.parse_args()
    print(args)

    torch.backends.cudnn.benchmark = True
    run(args)


if __name__ == "__main__":
    main()
