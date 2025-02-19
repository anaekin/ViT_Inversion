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
            self.r_feature = ((mean, var), (mean_bn, var_bn))
        elif self.model_type == "vit":
            # ViT feature regularization (mean & variance over token embeddings)
            embeddings = output  # Otherwise, the output itself is the embeddings
            mean = embeddings.mean(dim=[0, 1])  # Mean over batch & tokens
            var = embeddings.var(
                dim=[0, 1], unbiased=False
            )  # Variance over batch & tokens
            self.r_feature = (mean, var)

    def close(self):
        self.hook.remove()


def get_r_feature_loss(r_feature):
    """
    Given img_stats and target_stats (each a tuple: (mean, var)),
    return a scalar loss computed as the L2 distance between the means and variances.
    """
    (mean_img, var_img), (mean_bn, var_bn) = r_feature
    return torch.norm(mean_img - mean_bn, 2) + torch.norm(var_img - var_bn, 2)


def get_kl_loss(loss_r_feature_layer, target_stats):
    """
    Computes KL divergence loss between feature statistics from CNN BatchNorm layers
    and generated noise statistics.
    """
    eps = 1e-6  # Small value for numerical stability

    kl_losses = []
    for idx, hook in enumerate(loss_r_feature_layer):
        mean_gen, var_gen = hook.r_feature
        mean_bn, var_bn = target_stats[idx]
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


def get_extra_prior_losses(inputs_jit, bs):
    loss_l2 = torch.norm(inputs_jit.view(bs, -1), dim=1).mean()

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

    return loss_l2, loss_var_l1, loss_var_l2


def get_patch_prior_loss(inputs, patch_size=16):
    """
    Implements the GradViT Patch Prior Loss:
    Enforces smooth transitions between adjacent patches
    in vertical and horizontal directions.

    Args:
        inputs (torch.Tensor): Generated images (B, C, H, W).
        patch_size (int): Size of each ViT patch.

    Returns:
        torch.Tensor: Patch prior loss value.
    """
    B, C, H, W = inputs.shape

    # Ensure dimensions are divisible by patch size
    assert (
        H % patch_size == 0 and W % patch_size == 0
    ), "Image size must be divisible by patch size"

    vertical_loss = 0.0
    horizontal_loss = 0.0

    # Number of patches along height and width
    h_patches = H // patch_size
    w_patches = W // patch_size

    # Compute vertical patch differences
    for k in range(1, h_patches):
        upper_patch = inputs[:, :, (k - 1) * patch_size : k * patch_size, :]
        lower_patch = inputs[:, :, k * patch_size : (k + 1) * patch_size, :]
        vertical_loss += (upper_patch - lower_patch).pow(2).mean()

    # Compute horizontal patch differences
    for k in range(1, w_patches):
        left_patch = inputs[:, :, :, (k - 1) * patch_size : k * patch_size]
        right_patch = inputs[:, :, :, k * patch_size : (k + 1) * patch_size]
        horizontal_loss += (left_patch - right_patch).pow(2).mean()

    total_patch_loss = vertical_loss + horizontal_loss
    return total_patch_loss


class ViTDeepInversionClass(object):
    def __init__(
        self,
        bs=84,
        use_fp16=True,
        vit_teacher=None,
        net_guide=None,
        n_iterations=3000,
        store_dir="./generations/",
        parameters=dict(),
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
        self.n_iterations = n_iterations
        self.bs = bs
        self.use_fp16 = use_fp16
        self.save_every = 100
        self.jitter = jitter
        self.criterion = criterion
        self.network_output_function = network_output_function
        self.hook_for_display = hook_for_display

        if "resolution" in parameters.keys():
            self.image_resolution = parameters["resolution"]
            self.random_label = parameters["random_label"]
            self.start_noise = parameters["start_noise"]
            self.do_flip = parameters["do_flip"]
            self.store_best_images = parameters["store_best_images"]
        else:
            print("Provide a parameter dictionary")

        if "r_feature_scale" in coefficients.keys():
            self.vit_feature_loss = coefficients["vit_feature_loss"]
            self.lr = coefficients["lr"]
            self.warmup_length = coefficients["warmup_length"]
            self.first_bn_multiplier = coefficients["first_bn_multiplier"]
            self.tv_l1_scale = coefficients["tv_l1_scale"]
            self.tv_l2_scale = coefficients["tv_l2_scale"]
            self.l2_scale = coefficients["l2_scale"]
            self.patch_prior_scale = coefficients["patch_prior_scale"]
            self.extra_prior_scale = coefficients["extra_prior_scale"]
            self.image_prior_scale = coefficients["image_prior_scale"]
            self.r_feature_scale = coefficients["r_feature_scale"]
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

        # print([name for name, module in self.vit_teacher.named_modules()])
        hook_layers = [
            ("encoder.ln", "Final LayerNorm"),
            ("ln_2", "LayerNorm ln_2"),
            ("mlp.0", "MLP Layer"),
            ("mlp.3", "MLP Layer"),
            ("conv_proj", "Patch Embedding Layer"),
        ]

        for layer_name, description in hook_layers:
            for name, module in self.vit_teacher.named_modules():

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

            if not self.random_label:
                start = 151
                end = 268
                total = end - start
                mul = int(self.bs / total)
                rem = self.bs % total
                targets = [i for i in range(start, end)] * mul + [
                    i for i in range(start, start + rem)
                ]
                targets = torch.LongTensor(targets).to("cuda")
            else:
                targets = torch.LongTensor(
                    [random.randint(0, 999) for _ in range(self.bs)]
                ).to("cuda")

        print("Targets: ", targets)
        img_original = self.image_resolution
        data_type = torch.half if use_fp16 else torch.float
        inputs = torch.randn(
            (self.bs, 3, img_original, img_original),
            requires_grad=True,
            device="cuda",
            dtype=data_type,
        )
        pooling_function = nn.Identity()

        # Precompute target feature statistics from real dog images:
        target_r_feature_stats = self.get_target_r_feature_stats()
        print("Target r_feature stats computed from real dog images.")

        iteration = 0
        for lr_it, lower_res in enumerate([1]):
            scaler = torch.amp.GradScaler(device="cuda")
            lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res

            optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps=1e-8)
            do_clip = True

            lr_scheduler = lr_cosine_policy(
                self.lr, self.warmup_length, self.n_iterations
            )

            print(f"Running for {self.n_iterations} iterations...")
            for iteration_loc in range(self.n_iterations):
                iteration += 1
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                inputs_jit = pooling_function(inputs)
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

                    # CCE loss (Cross-Entropy loss)
                    loss_criterion = criterion(outputs, targets)

                    # R-feature loss (features from CNN)
                    # OR also called Image prior loss (from GradViT paper)
                    # rescale = [self.first_bn_multiplier] + [
                    #     1.0 for _ in range(len(self.loss_r_feature_layers_verifier) - 1)
                    # ]
                    rescale = [self.first_bn_multiplier] + [
                        1.0 + (self.first_bn_multiplier / (_s + 2))
                        for _s in range(len(self.loss_r_feature_layers_verifier) - 1)
                    ]
                    loss_r_feature = sum(
                        [
                            get_r_feature_loss(mod.r_feature) * rescale[idx]
                            for (idx, mod) in enumerate(
                                self.loss_r_feature_layers_verifier
                            )
                        ]
                    )

                    if self.vit_feature_loss == "l2":
                        loss_image_prior = sum(
                            get_r_feature_loss(
                                (mod.r_feature, target_r_feature_stats[idx])
                            )
                            for idx, mod in enumerate(self.loss_r_feature_layers)
                        )
                    elif self.vit_feature_loss == "kl":
                        loss_image_prior = get_kl_loss(
                            self.loss_r_feature_layers, target_r_feature_stats
                        )
                    else:
                        loss_image_prior = 0.0

                    # Get extra prior losses (from GradViT paper) - L2, TV_L1, TV_L2
                    loss_l2, loss_tv_l1, loss_tv_l2 = get_extra_prior_losses(
                        inputs_jit, bs=self.bs
                    )
                    loss_extra_prior = (
                        self.tv_l1_scale * loss_tv_l1
                        + self.tv_l2_scale * loss_tv_l2
                        + self.l2_scale * loss_l2
                    )

                    # Patch Prior Loss (From GradViT paper)
                    loss_patch_prior = get_patch_prior_loss(inputs.detach())

                    loss_aux = (
                        self.extra_prior_scale * loss_extra_prior
                        + self.patch_prior_scale * loss_patch_prior
                    )

                    loss = (
                        self.main_loss_scale * loss_criterion
                        + self.r_feature_scale * loss_r_feature
                        + self.image_prior_scale * loss_image_prior
                        + loss_aux
                    )

                if iteration % save_every == 0:
                    print("------------iteration {}----------".format(iteration))
                    print("loss_criterion", loss_criterion.item())
                    print("loss_r_feature", loss_r_feature.item())
                    print("loss_image_prior", loss_image_prior.item())
                    print("loss_aux", loss_aux.item())
                    print("     loss_extra_prior", loss_extra_prior.item())
                    print("         loss_tv_l1", loss_tv_l1.item())
                    print("         loss_tv_l2", loss_tv_l2.item())
                    print("         loss_l2", loss_l2.item())
                    print("     loss_patch_prior", loss_patch_prior.item())
                    print("loss", loss.item())

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
