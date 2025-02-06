# --------------------------------------------------------\
# Inversion of Vision Transformers (ViTs) using DeepInversion
# Animesh Jain
# --------------------------------------------------------

from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.optim as optim
import collections
import torch.cuda.amp as amp
import random
import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np

from utils.utils import (
    lr_cosine_policy,
    lr_policy,
    beta_policy,
    mom_cosine_policy,
    clip,
    denormalize,
    create_folder,
)


class DeepInversionFeatureHook:
    """
    DeepInversion feature hook for CNNs and ViTs.
    Tracks feature statistics for CNNs using BatchNorm and adapts for ViTs.
    """

    def __init__(self, module, model_type):
        self.model_type = model_type
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # (ANIMESH) Added conditions for CNN and ViT
        if self.model_type == "cnn":
            # CNN feature regularization (using BatchNorm statistics)
            nch = input[0].shape[1]
            mean = input[0].mean([0, 2, 3])
            var = (
                input[0]
                .permute(1, 0, 2, 3)
                .contiguous()
                .view([nch, -1])
                .var(1, unbiased=False)
            )
            # r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            #     module.running_mean.data - mean, 2
            # )

            # Extract BatchNorm's running statistics
            mean_bn = module.running_mean.data
            var_bn = module.running_var.data

            # Store values for later loss computation
            self.r_feature = (mean, var, mean_bn, var_bn)

        elif self.model_type == "vit":
            # ViT feature regularization (mean & variance over token embeddings)
            embeddings = output  # Shape: (B, num_tokens, D)
            mean = embeddings.mean(dim=[0, 1])  # Mean over batch & tokens
            var = embeddings.var(
                dim=[0, 1], unbiased=False
            )  # Variance over batch & tokens
            self.r_feature = torch.norm(mean, 2) + torch.norm(var, 2)

    def close(self):
        self.hook.remove()


def get_kl_loss(loss_r_feature_layers_verifier):
    """
    Computes KL divergence loss between feature statistics from CNN BatchNorm layers
    and generated noise statistics.
    """
    loss_kl = 0
    eps = 1e-6  # Small value for numerical stability

    for hook in loss_r_feature_layers_verifier:
        mean_gen, var_gen, mean_bn, var_bn = hook.r_feature

        # Ensure variance is positive to avoid numerical instability
        var_gen = torch.clamp(var_gen, min=eps)
        var_bn = torch.clamp(var_bn, min=eps)

        # Compute KL divergence
        kl_loss = 0.5 * (
            (var_gen / var_bn)
            + ((mean_bn - mean_gen) ** 2) / var_bn
            - 1
            + torch.log(var_bn / var_gen)
        )

        loss_kl += kl_loss.sum()  # Summing over channels

    return loss_kl / len(
        loss_r_feature_layers_verifier
    )  # Normalize by the number of layers


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = (
        torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    )
    loss_var_l1 = (
        (diff1.abs() / 255.0).mean()
        + (diff2.abs() / 255.0).mean()
        + (diff3.abs() / 255.0).mean()
        + (diff4.abs() / 255.0).mean()
    )
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


class ViTDeepInversionClass(object):
    def __init__(
        self,
        bs=84,
        use_fp16=True,
        net_teacher=None,
        net_verifier=None,
        path="./generations/",
        parameters=dict(),
        setting_id=0,
        jitter=30,
        criterion=None,
        coefficients=dict(),
        network_output_function=lambda x: x,
        hook_for_display=None,
    ):
        """
        :param bs: batch size per GPU for image generation
        :param use_fp16: use FP16 (or APEX AMP) for model inversion, uses less memory and is faster for GPUs with Tensor Cores
        :parameter net_teacher: Pytorch model to be inverted
        :param path: path where to write temporal images and data
        :param final_data_path: path to write final images into
        :param parameters: a dictionary of control parameters:
            "resolution": input image resolution, single value, assumed to be a square, 224
            "random_label" : for classification initialize target to be random values
            "start_noise" : start from noise, def True, other options are not supported at this time
        :param setting_id: predefined settings for optimization:
            0 - will run low resolution optimization for 1k and then full resolution for 1k;
            1 - will run optimization on high resolution for 2k
            2 - will run optimization on high resolution for 20k

        :param jitter: amount of random shift applied to image at every iteration
        :param coefficients: dictionary with parameters and coefficients for optimization.
            keys:
            "r_feature" - coefficient for feature distribution regularization
            "tv_l1" - coefficient for total variation L1 loss
            "tv_l2" - coefficient for total variation L2 loss
            "l2" - l2 penalization weight
            "lr" - learning rate for optimization
            "main_loss_multiplier" - coefficient for the main loss optimization
            "kl_loss_scale" - coefficient for Adaptive DeepInversion, competition, def =0 means no competition
        network_output_function: function to be applied to the output of the network to get the output
        hook_for_display: function to be executed at every print/save call, useful to check accuracy of verifier
        """

        print("Deep inversion class generation")
        # for reproducibility
        torch.manual_seed(torch.cuda.current_device())

        self.net_teacher = net_teacher
        self.net_verifier = net_verifier

        if "resolution" in parameters.keys():
            self.image_resolution = parameters["resolution"]
            self.random_label = parameters["random_label"]
            self.start_noise = parameters["start_noise"]
            self.do_flip = parameters["do_flip"]
            self.store_best_images = parameters["store_best_images"]
        else:
            print("Provide a parameter dictionary")

        self.setting_id = setting_id
        self.bs = bs  # batch size
        self.use_fp16 = use_fp16
        self.save_every = 100
        self.jitter = jitter
        self.criterion = criterion
        self.network_output_function = network_output_function
        self.hook_for_display = hook_for_display

        if "r_feature_scale" in coefficients.keys():
            self.ln_reg_scale = coefficients["r_feature_scale"]
            self.first_ln_multiplier = coefficients["first_ln_multiplier"]
            self.var_l1_scale = coefficients["tv_l1_scale"]
            self.var_l2_scale = coefficients["tv_l2_scale"]
            self.l2_scale = coefficients["l2_scale"]
            self.kl_loss_scale = coefficients["kl_loss_scale"]
            self.lr = coefficients["lr"]
            self.main_loss_multiplier = coefficients["main_loss_multiplier"]
        else:
            print("Provide a coefficient dictionary")

        self.num_generations = 0

        prefix = path
        self.prefix = prefix

        local_rank = torch.cuda.current_device()
        if local_rank == 0:
            create_folder(prefix)
            create_folder(prefix + "/best_images/")
            create_folder(prefix + "/final_images/")

        ## Create hooks for feature statistics
        self.loss_r_feature_layers = []
        self.loss_r_feature_layers_verifier = []

        # (ANIMESH) Added support for ViT
        # Adding hooks for feature statistics
        for module in self.net_teacher.modules():
            if isinstance(module, nn.LayerNorm):
                # Hook for ViT LayerNorm layers
                self.loss_r_feature_layers.append(
                    DeepInversionFeatureHook(module, model_type="vit")
                )

        for module in self.net_verifier.modules():
            if isinstance(module, nn.BatchNorm2d):
                # Hook for CNN BatchNorm layers
                self.loss_r_feature_layers_verifier.append(
                    DeepInversionFeatureHook(module, model_type="cnn")
                )

        print("Created a total of {} hooks".format(len(self.loss_r_feature_layers)))
        print(
            "Created a total of {} verifier hooks".format(
                len(self.loss_r_feature_layers_verifier)
            )
        )

    def get_images(self, targets=None):
        print("Generating images...")

        net_teacher = self.net_teacher
        net_verifier = self.net_verifier
        use_fp16 = self.use_fp16
        save_every = self.save_every

        # kl_loss = nn.KLDivLoss(reduction="batchmean").cuda()
        local_rank = torch.cuda.current_device()
        best_cost = 1e4
        criterion = self.criterion

        # setup target labels
        if targets is None:
            # only works for classification now, for other tasks need to provide target vector
            targets = torch.LongTensor(
                [random.randint(0, 999) for _ in range(self.bs)]
            ).to("cuda")
            if not self.random_label:
                # preselected classes, good for ResNet50v1.5
                targets = [
                    153,
                    200,
                    229,
                    230,
                    235,
                    238,
                    239,
                    245,
                    248,
                    251,
                    252,
                    254,
                    256,
                    275,
                    537,
                    934,  # Hot dog
                ]

                print("Batch size: ", self.bs)
                print("# of targets: ", len(targets))

                targets = torch.LongTensor(targets * (int(self.bs / len(targets)))).to(
                    "cuda"
                )

        img_original = self.image_resolution

        data_type = torch.half if use_fp16 else torch.float
        inputs = torch.randn(
            (self.bs, 3, img_original, img_original),
            requires_grad=True,
            device="cuda",
            dtype=data_type,
        )
        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)

        if self.setting_id == 0:
            skipfirst = False
        else:
            skipfirst = True

        iteration = 0
        for lr_it, lower_res in enumerate([2, 1]):
            scaler = torch.amp.GradScaler(device="cuda")

            if lr_it == 0:
                iterations_per_layer = 2000
            else:
                iterations_per_layer = 1000 if not skipfirst else 2000
                if self.setting_id == 2:
                    iterations_per_layer = 20000

            if lr_it == 0 and skipfirst:
                continue

            lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res

            if self.setting_id == 0:
                # multi resolution, 2k iterations with low resolution, 1k at normal, ResNet50v1.5 works the best, ResNet50 is ok
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps=1e-8)
                do_clip = True
            elif self.setting_id == 1:
                # 2k normal resolultion, for ResNet50v1.5; Resnet50 works as well
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps=1e-8)
                do_clip = True
            elif self.setting_id == 2:
                # 20k normal resolution the closes to the paper experiments for ResNet50
                optimizer = optim.Adam(
                    [inputs], lr=self.lr, betas=[0.9, 0.999], eps=1e-8
                )
                do_clip = False

            # if use_fp16:
            #     static_loss_scale = 256
            #     static_loss_scale = "dynamic"
            #     _, optimizer = amp.initialize(
            #         [], optimizer, opt_level="O2", loss_scale=static_loss_scale
            #     )

            lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                # learning rate scheduling
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                # perform downsampling if needed
                # if lower_res != 1:
                #     inputs_jit = pooling_function(inputs)
                # else:
                inputs_jit = inputs

                # apply random jitter offsets
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

                # Flipping
                flip = random.random() > 0.5
                if flip and self.do_flip:
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))

                # forward pass
                optimizer.zero_grad()
                net_teacher.zero_grad()
                net_verifier.zero_grad()

                with torch.autocast(device_type="cuda", dtype=data_type):
                    outputs = net_teacher(inputs_jit)
                    # outputs = self.network_output_function(outputs)

                    ver_outputs = net_verifier(inputs_jit)

                    # R_cross classification loss
                    # (ANIMESH) Loss 1: CCE
                    loss_criterion = criterion(outputs, targets)

                    # R_prior losses
                    loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

                    # R_feature loss
                    # (ANIMESH) Loss 2: Feature distribution regularization
                    # rescale = [self.first_bn_multiplier] + [
                    #     1.0 for _ in range(len(self.loss_r_feature_layers) - 1)
                    # ]
                    rescale_vit = [self.first_ln_multiplier] + [
                        1.0 for _ in range(len(self.loss_r_feature_layers) - 1)
                    ]
                    # loss_r_feature = sum(
                    #     [
                    #         mod.r_feature * rescale[idx]
                    #         for (idx, mod) in enumerate(self.loss_r_feature_layers)
                    #     ]
                    # )

                    loss_r_feature = sum(
                        [
                            mod.r_feature * rescale_vit[idx]
                            for (idx, mod) in enumerate(self.loss_r_feature_layers)
                        ]
                    )

                    # KL divergence loss
                    # (ANIMESH) Loss 3: KL divergence loss
                    loss_kl = get_kl_loss(self.loss_r_feature_layers_verifier)

                    # l2 loss on images
                    loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()

                    # combining losses
                    loss_aux = (
                        self.var_l2_scale * loss_var_l2
                        + self.var_l1_scale * loss_var_l1
                        + self.ln_reg_scale * loss_r_feature
                        + self.l2_scale * loss_l2
                        + self.kl_loss_scale * loss_kl
                    )

                    loss = self.main_loss_multiplier * loss_criterion + loss_aux

                if local_rank == 0:
                    if iteration % save_every == 0:
                        print("------------iteration {}----------".format(iteration))

                        print("criterion_loss", loss_criterion.item())
                        print("loss_var_l1", loss_var_l1.item())
                        print("loss_var_l2", loss_var_l2.item())
                        print("loss_l2", loss_l2.item())
                        print("loss_kl", loss_kl.item())
                        print("loss_r_feature", loss_r_feature.item())
                        print("loss_aux", loss_aux.item())
                        print("total loss", loss.item())

                        if self.hook_for_display is not None:
                            self.hook_for_display(inputs, targets)

                # do image update
                if use_fp16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # clip color outlayers
                if do_clip:
                    inputs.data = clip(inputs.data, use_fp16=use_fp16)

                if best_cost > loss.item() or iteration == 1:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()

                if iteration % save_every == 0 and (save_every > 0):
                    if local_rank == 0:
                        vutils.save_image(
                            inputs,
                            "{}/best_images/output_{:05d}_gpu_{}.png".format(
                                self.prefix, iteration // save_every, local_rank
                            ),
                            normalize=True,
                            scale_each=True,
                            nrow=int(10),
                        )

        if self.store_best_images:
            best_inputs = denormalize(best_inputs)
            self.save_images(best_inputs, targets)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)

    def save_images(self, images, targets):
        # method to store generated images locally
        local_rank = torch.cuda.current_device()
        for id in range(images.shape[0]):
            class_id = targets[id].item()
            if 0:
                # save into separate folders
                place_to_store = "{}/s{:03d}/img_{:05d}_id{:03d}_gpu_{}_2.jpg".format(
                    self.final_data_path, class_id, self.num_generations, id, local_rank
                )
            else:
                place_to_store = "{}/img_s{:03d}_{:05d}_id{:03d}_gpu_{}_2.jpg".format(
                    self.final_data_path, class_id, self.num_generations, id, local_rank
                )

            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)

    def generate_batch(self, targets=None):
        # Put to eval mode
        net_teacher = self.net_teacher
        net_verifier = self.net_verifier

        use_fp16 = self.use_fp16

        if targets is not None:
            targets = torch.from_numpy(np.array(targets).squeeze()).cuda()
            if use_fp16:
                targets = targets.half()

        self.get_images(targets=targets)

        net_teacher.eval()
        net_verifier.eval()

        self.num_generations += 1
