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

random.seed(1)


class AccuracyTracker:
    def __init__(self):
        self.current_accuracy = 0.0

    def update_accuracy(self, acc):
        self.current_accuracy = acc


acc_tracker = AccuracyTracker()


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

    acc_tracker.update_accuracy(prec1.item())

    print("Verifier accuracy (top 1): ", prec1.item())
    print("Verifier accuracy (top 5): ", prec5.item())


# Command to run the script:
# python run.py --do_flip --r_feature_scale=0.0 --image_prior_scale=0.0 --n_iterations=5000 --lr=0.25 --l2_scale=0.0 --tv_l1_scale=0.0 --tv_l2_scale=0.0 --main_loss_scale=1.0 --image_prior_scale=0.0 --store_best_images


def run(args):
    local_rank = args["local_rank"]
    no_cuda = args["no_cuda"]
    use_fp16 = args["fp16"]

    bs = args["bs"]
    n_iterations = args["n_iterations"]
    jitter = args["jitter"]

    store_dir = args["store_dir"]
    store_dir = "./generations/{}".format(store_dir)

    parameters = dict()
    parameters["resolution"] = 224
    parameters["start_noise"] = True
    parameters["random_label"] = args["random_label"]
    parameters["do_flip"] = args["do_flip"]
    parameters["store_best_images"] = args["store_best_images"]

    coefficients = dict()
    coefficients["vit_feature_loss"] = args["vit_feature_loss"]
    coefficients["lr"] = args["lr"]
    coefficients["warmup_length"] = args["warmup_length"]
    coefficients["first_bn_multiplier"] = args["first_bn_multiplier"]
    coefficients["main_loss_scale"] = args["main_loss_scale"]
    coefficients["image_prior_scale"] = args["image_prior_scale"]
    coefficients["r_feature_scale"] = args["r_feature_scale"]
    coefficients["tv_l1_scale"] = args["tv_l1_scale"]
    coefficients["tv_l2_scale"] = args["tv_l2_scale"]
    coefficients["l2_scale"] = args["l2_scale"]
    coefficients["patch_prior_scale"] = args["patch_prior_scale"]
    coefficients["extra_prior_scale"] = args["extra_prior_scale"]

    torch.manual_seed(local_rank)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    )
    torch.cuda.set_device(local_rank)

    print("Loading model for inversion...")

    ### load models
    # ! Load from torchvision directly
    vit = torchvision.models.vit_b_16(weights="IMAGENET1K_V1").to(device)

    if use_fp16:
        vit.half()

    for module in vit.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.float()

    vit.eval()

    # reserved to compute test accuracy on generated images by different networks
    net_guide = None

    print("Loading guiding CNN...")

    # Load ResNet-50
    net_guide = torchvision.models.resnet50(weights="IMAGENET1K_V1").to(device)

    if use_fp16:
        # Convert model to FP16
        net_guide.half()

        # Ensure BatchNorm stays in FP32 for stability
        for module in net_guide.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.float()

    # Set model to evaluation mode
    net_guide.eval()

    net_verifier = None
    print("Loading verifier...", args["verifier_arch"])
    net_verifier = models.__dict__[args["verifier_arch"]](pretrained=True).to(device)
    net_verifier.eval()

    if use_fp16:
        net_verifier = net_verifier.half()

    # from deepinversion_vit import ViTDeepInversionClass
    from deepinversion_vit_combined import ViTDeepInversionClass

    criterion = nn.CrossEntropyLoss()

    ViTDeepInversionEngine = ViTDeepInversionClass(
        vit_teacher=vit,
        net_guide=net_guide,
        store_dir=store_dir,
        n_iterations=n_iterations,
        bs=bs,
        use_fp16=use_fp16,
        jitter=jitter,
        criterion=criterion,
        parameters=parameters,
        coefficients=coefficients,
        network_output_function=lambda x: x,
        hook_for_display=lambda x, y: validate(x, y, net_verifier, use_fp16),
    )
    ViTDeepInversionEngine.generate_batch()  # Runs iterations and saves accuracy

    # Return final recorded accuracy
    return acc_tracker.current_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        "--rank",
        type=int,
        default=1,
        help="Rank of the current process.",
    )
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--bs", default=32, type=int, help="batch size")
    parser.add_argument(
        "--n_iterations", default=3000, type=int, help="Number of iterations"
    )
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

    parser.add_argument(
        "--verifier_arch",
        type=str,
        default="mobilenet_v2",
        help="arch name from torchvision models to act as a verifier",
    )

    # Coefficients for optimization
    parser.add_argument(
        "--vit_feature_loss",
        type=str,
        default="l2",
        help="Use L2-norm or KL div for image prior",
    )
    parser.add_argument(
        "--lr", type=float, default=0.25, help="learning rate for optimization"
    )
    parser.add_argument(
        "--warmup_length",
        type=float,
        default=100,
        help="warmup length for cosine scheduler learning rate",
    )
    parser.add_argument(
        "--first_bn_multiplier",
        type=float,
        default=10.0,
        help="additional multiplier on first bn layer of R_feature",
    )
    parser.add_argument(
        "--main_loss_scale",
        type=float,
        default=0.05,
        help="coefficient for the main loss in optimization",
    )
    parser.add_argument(
        "--r_feature_scale",
        type=float,
        default=0.005,
        help="coefficient for feature distribution regularization",
    )
    parser.add_argument(
        "--image_prior_scale",
        type=float,
        default=0.0001,
        help="Coefficient for KL Loss",
    )
    parser.add_argument(
        "--patch_prior_scale",
        type=float,
        default=0.00001,
        help="coefficient for the patch prior loss",
    )
    parser.add_argument(
        "--extra_prior_scale",
        type=float,
        default=1.0,
        help="coefficient for the patch prior loss",
    )
    parser.add_argument(
        "--tv_l1_scale",
        type=float,
        default=0.0001,
        help="coefficient for total variation L1 loss",
    )
    parser.add_argument(
        "--tv_l2_scale",
        type=float,
        default=0.000001,
        help="coefficient for total variation L2 loss",
    )
    parser.add_argument(
        "--l2_scale", type=float, default=0.000001, help="l2 loss on the image"
    )
    args = parser.parse_args()
    print(args)

    torch.backends.cudnn.benchmark = True
    params = vars(args)  # transforms argparse namespace to dict
    final_acc = run(params)
    print("Final verifier accuracy (returned):", final_acc)


if __name__ == "__main__":
    main()
