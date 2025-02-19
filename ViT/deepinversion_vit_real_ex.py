from __future__ import division, print_function, absolute_import, unicode_literals

import torch
import torch.nn as nn
import torch.optim as optim
import collections
import torch.cuda.amp as amp
import random
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
        self.r_feature = None

    def hook_fn(self, module, input, output):
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
            # Extract BatchNorm's running statistics
            mean_bn = module.running_mean.data
            var_bn = module.running_var.data
            self.r_feature = (mean, var, mean_bn, var_bn)
        elif self.model_type == "vit":
            # ViT feature regularization (mean & variance over token embeddings)
            if isinstance(output, tuple):
                embeddings = output[
                    0
                ]  # Attention output is typically the first element
            else:
                embeddings = output  # Otherwise, the output itself is the embeddings
            mean = embeddings.mean(dim=[0, 1])  # Mean over batch & tokens
            var = embeddings.var(
                dim=[0, 1], unbiased=False
            )  # Variance over batch & tokens
            self.r_feature = (mean, var)

    def close(self):
        self.hook.remove()


def get_kl_loss(loss_r_feature_layers_verifier):
    """
    Computes KL divergence loss between feature statistics from CNN BatchNorm layers
    and generated noise statistics.
    """
    total_kl_loss = 0
    eps = 1e-6  # Small value for numerical stability

    kl_losses = []
    for hook in loss_r_feature_layers_verifier:
        mean_gen, var_gen, mean_bn, var_bn = hook.r_feature
        var_gen = torch.clamp(var_gen, min=eps)
        var_bn = torch.clamp(var_bn, min=eps)
        kl_loss = 0.5 * (
            torch.log(var_bn / var_gen)
            + ((var_gen + (mean_bn - mean_gen) ** 2) / var_bn)
            - 1
        )
        kl_losses.append(kl_loss.mean())
    normalized_kl_loss = torch.stack(kl_losses).mean() if kl_losses else 0
    return normalized_kl_loss


def get_image_prior_losses(inputs_jit):
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


def compute_target_r_feature_loss(gen_stats, target_stats):
    """
    Given gen_stats and target_stats (each a tuple: (mean, var)),
    return a scalar loss computed as the L2 distance between the means and variances.
    """
    mean_gen, var_gen = gen_stats
    mean_target, var_target = target_stats
    return torch.norm(mean_gen - mean_target, 2) + torch.norm(var_gen - var_target, 2)


class ViTDeepInversionClass(object):
    def __init__(
        self,
        bs=84,
        use_fp16=True,
        vit_teacher=None,
        net_guide=None,
        store_dir="./generations/",
        parameters=dict(),
        setting_id=0,
        jitter=30,
        criterion=None,
        coefficients=dict(),
        network_output_function=lambda x: x,
        hook_for_display=None,
    ):
        print("ViT Deep inversion class generation")
        self.local_rank = torch.cuda.current_device()
        torch.manual_seed(self.local_rank)

        self.vit_teacher = vit_teacher
        self.net_guide = net_guide

        if "resolution" in parameters.keys():
            self.image_resolution = parameters["resolution"]
            self.random_label = parameters["random_label"]
            self.start_noise = parameters["start_noise"]
            self.do_flip = parameters["do_flip"]
            self.store_best_images = parameters["store_best_images"]
        else:
            print("Provide a parameter dictionary")

        self.setting_id = setting_id
        self.bs = bs
        self.use_fp16 = use_fp16
        self.save_every = 100
        self.jitter = jitter
        self.criterion = criterion
        self.network_output_function = network_output_function
        self.hook_for_display = hook_for_display

        if "r_feature_scale" in coefficients.keys():
            self.lr = coefficients["lr"]
            self.ln_reg_scale = coefficients["r_feature_scale"]
            self.first_ln_multiplier = coefficients["first_ln_multiplier"]
            self.var_l1_scale = coefficients["tv_l1_scale"]
            self.var_l2_scale = coefficients["tv_l2_scale"]
            self.l2_scale = coefficients["l2_scale"]
            self.kl_loss_scale = coefficients["kl_loss_scale"]
            self.main_loss_scale = coefficients["main_loss_scale"]
        else:
            print("Provide a coefficient dictionary")

        self.num_generations = 0

        self.prefix = store_dir
        self.generated_images_path = self.prefix + "/best_images/"
        self.final_images_path = self.prefix + "/final_images/"

        create_folder(self.prefix)
        create_folder(self.generated_images_path)
        create_folder(self.final_images_path)

        ## Create hooks for feature statistics
        self.loss_r_feature_layers = []
        self.loss_r_feature_layers_verifier = []

        for name, module in self.vit_teacher.named_modules():
            hook_layers = [
                ("encoder.ln", "Final LayerNorm"),
                ("ln_2", "LayerNorm ln_2"),
                ("mlp.0", "MLP Layer"),
                ("mlp.3", "MLP Layer"),
                ("conv_proj", "Patch Embedding Layer"),
            ]

            for layer_name, description in hook_layers:
                if layer_name in name:
                    self.loss_r_feature_layers.append(
                        DeepInversionFeatureHook(module, model_type="vit")
                    )
                    print(f"Hook added to {description}: {name}")

        for module in self.net_guide.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers_verifier.append(
                    DeepInversionFeatureHook(module, model_type="cnn")
                )

        print("Created a total of {} hooks".format(len(self.loss_r_feature_layers)))
        print(
            "Created a total of {} verifier hooks".format(
                len(self.loss_r_feature_layers_verifier)
            )
        )

    # Load a batch of real dog images
    def load_real_dog_images(self):
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader

        transform = transforms.Compose(
            [
                transforms.Resize((self.image_resolution, self.image_resolution)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # Here, we store the OxfordIIITPet dataset in "./data/OxfordIIITPet"
        dataset = datasets.OxfordIIITPet(
            root="./data/OxfordIIITPet",
            split="trainval",
            target_types="category",
            transform=transform,
            download=True,
        )

        # In OxfordIIITPet, dog categories have indices >=12 (since there are 37 categories: 0-11 cats, 12-36 dogs)
        dog_indices = [i for i in range(len(dataset)) if dataset[i][1] >= 12]
        from torch.utils.data import Subset

        dog_subset = Subset(dataset, dog_indices)
        dog_loader = DataLoader(
            dog_subset, batch_size=self.bs, shuffle=True, num_workers=1
        )

        return dog_loader

    # Extract real image feature statistics using the already attached hooks.
    def get_target_r_feature_stats(self):
        """
        Compute aggregated target feature statistics (mean and variance)
        for each hooked ViT layer using all batches from the real dog DataLoader.
        This version accumulates the statistics in lists and then averages them.
        """
        # Since self.load_real_dog_images() returns a DataLoader (dog_loader)
        dog_loader = self.load_real_dog_images()
        self.vit_teacher.eval()

        # Create lists to collect batch-wise means and variances for each hook.
        num_hooks = len(self.loss_r_feature_layers)
        all_means = [[] for _ in range(num_hooks)]
        all_vars = [[] for _ in range(num_hooks)]

        with torch.no_grad():
            for images, _ in dog_loader:
                images = images.to("cuda")
                _ = self.vit_teacher(
                    images
                )  # Forward pass updates each hook's r_feature
                for i, hook in enumerate(self.loss_r_feature_layers):
                    # Each hook.r_feature is expected to be a tuple (mean, var)
                    if hook.r_feature is None:
                        continue
                    batch_mean, batch_var = hook.r_feature
                    all_means[i].append(batch_mean)
                    all_vars[i].append(batch_var)

        # For each hook, stack the collected statistics and average them.
        target_stats = []
        for m, v in zip(all_means, all_vars):
            avg_mean = torch.stack(m, dim=0).mean(dim=0)
            avg_var = torch.stack(v, dim=0).mean(dim=0)
            target_stats.append((avg_mean, avg_var))

        return target_stats

    def get_images(self, targets=None):
        print("Generating images...")

        vit_teacher = self.vit_teacher
        net_guide = self.net_guide
        use_fp16 = self.use_fp16
        save_every = self.save_every
        local_rank = self.local_rank
        best_cost = 1e4
        criterion = self.criterion

        # Setup target labels
        if targets is None:
            targets = torch.LongTensor(
                [random.randint(0, 999) for _ in range(self.bs)]
            ).to("cuda")
            if not self.random_label:
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
                    239,
                ]
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

        # Precompute target feature statistics from real dog images:
        target_r_feature_stats = self.get_target_r_feature_stats()
        print("Target r_feature stats computed from real dog images.")

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
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps=1e-8)
                do_clip = True
            elif self.setting_id == 1:
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps=1e-8)
                do_clip = True
            elif self.setting_id == 2:
                optimizer = optim.Adam(
                    [inputs], lr=self.lr, betas=[0.9, 0.999], eps=1e-8
                )
                do_clip = False

            lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                inputs_jit = inputs
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

                flip = random.random() > 0.5
                if flip and self.do_flip:
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))

                optimizer.zero_grad()
                vit_teacher.zero_grad()
                net_guide.zero_grad()

                with torch.autocast(device_type="cuda", dtype=data_type):
                    outputs = vit_teacher(inputs_jit)
                    ver_outputs = net_guide(inputs_jit)

                    # CCE loss
                    loss_criterion = criterion(outputs, targets)

                    # R-feature loss
                    loss_r_feature = sum(
                        compute_target_r_feature_loss(
                            mod.r_feature, target_r_feature_stats[idx]
                        )
                        for idx, mod in enumerate(self.loss_r_feature_layers)
                    ) / len(self.loss_r_feature_layers)

                    # KL Loss - CNN guiding model
                    loss_kl = get_kl_loss(self.loss_r_feature_layers_verifier)

                    # Total variance and L2 loss
                    loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)
                    loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()

                    loss_aux = (
                        self.var_l2_scale * loss_var_l2
                        + self.var_l1_scale * loss_var_l1
                        + self.ln_reg_scale * loss_r_feature
                        + self.l2_scale * loss_l2
                        + self.kl_loss_scale * loss_kl
                    )

                    loss = self.main_loss_scale * loss_criterion + loss_aux

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

                if use_fp16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if do_clip:
                    inputs.data = clip(inputs.data, use_fp16=use_fp16)

                if best_cost > loss.item() or iteration == 1:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()

                if iteration % save_every == 0 and (save_every > 0):
                    vutils.save_image(
                        inputs,
                        "{}/best_images/output_{:05d}_gpu_{}.png".format(
                            self.prefix, iteration // save_every, local_rank
                        ),
                        normalize=True,
                        scale_each=True,
                        nrow=10,
                    )

        if self.store_best_images:
            best_inputs = denormalize(best_inputs)
            self.save_images(best_inputs, targets)

        optimizer.state = collections.defaultdict(dict)

    def save_images(self, images, targets):
        local_rank = self.local_rank
        for id in range(images.shape[0]):
            class_id = targets[id].item()
            place_to_store = "{}/img_s{:03d}_{:05d}_id{:03d}_gpu_{}_2.jpg".format(
                self.final_images_path, class_id, self.num_generations, id, local_rank
            )
            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)

    def generate_batch(self, targets=None):
        vit_teacher = self.vit_teacher
        use_fp16 = self.use_fp16
        if targets is not None:
            targets = torch.from_numpy(np.array(targets).squeeze()).cuda()
            if use_fp16:
                targets = targets.half()
        self.get_images(targets=targets)
        vit_teacher.eval()
        self.num_generations += 1
