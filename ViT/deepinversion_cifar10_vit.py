import argparse
import random
import torch
import torch.nn as nn

# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

# import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.transforms as transforms

import numpy as np
import os
import gc
import glob
import collections
from torchvision.datasets import CIFAR10
from torchvision.models import vit_b_16, resnet18, ViT_B_16_Weights
from torch.utils.data import DataLoader


# Vision Transformer (ViT) Model Wrapper
class ViTWrapper(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ViTWrapper, self).__init__()
        self.vit = vit_b_16(
            weights=ViT_B_16_Weights.DEFAULT
        ).cuda()  # Load pretrained ViT model and move to CUDA
        # Adjust the classifier head
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes).cuda()  # Move to CUDA

    def forward(self, x):
        return self.vit(x)


# DeepInversionFeatureHook
class DeepInversionFeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.r_feature = None

    def hook_fn(self, module, input, output):
        # Ensure input is a tuple and has at least one tensor
        if not input or len(input[0].shape) < 3:
            print(f"Invalid input shape in hook: {input[0].shape}")
            self.r_feature = 0  # Default to zero if shape is invalid
            return

        # Assuming input shape is (batch_size, num_tokens, embedding_dim)
        batch_size, num_tokens, embedding_dim = input[0].shape

        # Compute mean and variance across tokens (axis 1)
        mean = input[0].mean(dim=1)  # Mean across tokens
        var = input[0].var(dim=1, unbiased=False)  # Variance across tokens

        # Regularization loss for feature statistics based on computed mean and variance
        # Here, instead of using running stats, we just regularize using the mean/variance of the activations
        # You can adjust this based on your specific needs or leave as is
        r_feature = torch.norm(var, 2) + torch.norm(mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


# Function to Generate Images using Deep Inversion
def get_images_vit(
    vit,
    bs=64,
    epochs=1000,
    var_scale=0.00005,
    l2_coeff=0.0,
    competitive_scale=0.1,
    ln_reg_scale=0.0,
    img_size=224,
    debug_output=True,
    net_student=None,
    prefix=None,
    batch_idx=-1,
    inputs=None,
):
    """
    Generate images using Deep Inversion with Vision Transformer (ViT).
    """
    criterion = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean").cuda()
    optimizer = optim.Adam([inputs], lr=0.1)

    inputs.data = torch.randn(
        (bs, 3, img_size, img_size), requires_grad=True, device="cuda"
    )

    optimizer.state = collections.defaultdict(dict)

    # preventing backpropagation through student for Adaptive DeepInversion
    net_student.eval()
    best_cost = 1e6

    # Get a batch of target labels from the dog dataset
    targets = torch.LongTensor([5 for _ in range(bs)]).to("cuda")

    # Register hooks on layers of ViT
    loss_r_feature_layers = []
    for name, module in vit.vit.named_modules():
        if isinstance(
            module, nn.LayerNorm
        ):  # Example: Using LayerNorm layers for hooks
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    lim_0, lim_1 = 2, 2

    for epoch in range(epochs):
        # apply random jitter offsets
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))

        optimizer.zero_grad()
        vit.zero_grad()
        outputs = vit(inputs_jit)  # Forward pass
        loss = criterion(outputs, targets)  # Cross-entropy loss
        loss_target = loss.item()

        # Competitive loss: Jensen-Shannon divergence
        if competitive_scale != 0.0:
            net_student.zero_grad()
            outputs_student = net_student(inputs_jit)
            T = 3.0  # Temperature scaling

            P = F.softmax(outputs_student / T, dim=1)
            Q = F.softmax(outputs / T, dim=1)
            M = 0.5 * (P + Q)

            P = torch.clamp(P, 0.01, 0.99)
            Q = torch.clamp(Q, 0.01, 0.99)
            M = torch.clamp(M, 0.01, 0.99)

            eps = 0.0
            loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(
                torch.log(Q + eps), M
            )
            loss_verifier_cig = 1.0 - torch.clamp(
                loss_verifier_cig, 0.0, 1.0
            )  # JS divergence regularization

            loss += competitive_scale * loss_verifier_cig

        # Total variation loss for smoothness
        diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
        diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
        loss_var = (
            torch.norm(diff1)
            + torch.norm(diff2)
            + torch.norm(diff3)
            + torch.norm(diff4)
        )
        loss += var_scale * loss_var

        loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])
        loss = loss + ln_reg_scale * loss_distr  # best for noise before BN

        # L2 loss on inputs to prevent extreme pixel values
        loss = loss + l2_coeff * torch.norm(inputs_jit, 2)

        if debug_output and epoch % 200 == 0:
            print(
                f"It {epoch}\t Losses: total: {loss.item():3.3f},\ttarget: {loss_target:3.3f} \tR_feature_loss unscaled:\t {loss_distr.item():3.3f}"
            )
            vutils.save_image(
                inputs.data.clone(),
                "./{}/output_{}.png".format(prefix, epoch // 200),
                normalize=True,
                scale_each=True,
                nrow=10,
            )

        if best_cost > loss.item():
            best_cost = loss.item()
            best_inputs = inputs.data

        # backward pass
        loss.backward()

        optimizer.step()

    outputs = vit(best_inputs)
    _, predicted_teach = outputs.max(1)

    outputs_student = net_student(best_inputs)
    _, predicted_std = outputs_student.max(1)

    if batch_idx == 0:
        print(
            "Teacher correct out of {}: {}, loss at {}".format(
                bs,
                predicted_teach.eq(targets).sum().item(),
                criterion(outputs, targets).item(),
            )
        )
        print(
            "Student correct out of {}: {}, loss at {}".format(
                bs,
                predicted_std.eq(targets).sum().item(),
                criterion(outputs_student, targets).item(),
            )
        )

    name_use = "best_images"
    if prefix is not None:
        name_use = prefix + name_use
    next_batch = len(glob.glob("./%s/*.png" % name_use)) // 1

    vutils.save_image(
        best_inputs[:20].clone(),
        "./{}/output_{}.png".format(name_use, next_batch),
        normalize=True,
        scale_each=True,
        nrow=10,
    )

    return best_inputs


def test(vit_test):
    """
    Test the accuracy of the teacher Vision Transformer (ViT) model on the test dataset.

    Args:
        net_teacher (nn.Module): The teacher ViT model.
        testloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the test (CPU or CUDA).
    """
    print("==> Teacher validation")
    vit_test.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()  # Define the loss function

    with torch.no_grad():  # Disable gradient computation for testing
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass through the teacher model
            outputs = vit_test(inputs)

            # Compute the loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # Get predicted class indices
            _, predicted = outputs.max(1)

            # Update totals
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Print the final results
    print(
        "Loss: %.3f | Acc: %.3f%% (%d/%d)"
        % (test_loss / (batch_idx + 1), 100.0 * correct / total, correct, total)
    )


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    fp16 = False
    bs = 128
    img_size = 224

    # Transform for CIFAR-10 images to match ViT input size (224x224)
    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),  # Resize to 224x224
    #         transforms.ToTensor(),  # Convert to tensor
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize
    #     ]
    # )

    # Load CIFAR-10 dataset with filtering for dog class only
    class DogOnlyDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = [
                (img, label) for img, label in dataset if label == 5
            ]  # Dog class label is 5

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            return self.dataset[idx]

    # Load pretrained ViT model as teacher
    vit_teacher = ViTWrapper(num_classes=10, pretrained=True)
    vit_teacher = vit_teacher.to(device)
    vit_teacher.eval()

    net_student = resnet18(pretrained=True)
    num_classes = 10  # Adjust this to 2 for binary classification
    in_features = net_student.fc.in_features
    net_student.fc = nn.Linear(in_features, num_classes)
    net_student = net_student.to(device)

    # place holder for inputs
    data_type = torch.half if fp16 else torch.float
    inputs = torch.randn(
        (bs, 3, img_size, img_size), requires_grad=True, device=device, dtype=data_type
    )

    batch_idx = 0
    prefix = "runs/data_generation/"

    for create_folder in [prefix, prefix + "/best_images/"]:
        if not os.path.exists(create_folder):
            os.makedirs(create_folder)

    cudnn.benchmark = True

    if 0:
        # loading
        # vit_test = ViTWrapper(num_classes=10, pretrained=True)
        # vit_test.to(device)
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        cifar10 = CIFAR10(
            root="./data/cifar10", train=True, download=True, transform=transform
        )
        # dog_only_dataset = DogOnlyDataset(cifar10)
        dataloader = DataLoader(cifar10, batch_size=bs, shuffle=True)
        # Checking teacher accuracy
        print("Checking teacher accuracy")
        test(vit_teacher)

    # Generate images using Deep Inversion
    inverted_images = get_images_vit(
        vit=vit_teacher,
        bs=bs,
        epochs=1000,
        var_scale=0.0001,
        ln_reg_scale=10,
        l2_coeff=0.0,
        competitive_scale=0.01,
        img_size=img_size,
        net_student=net_student,
        batch_idx=batch_idx,
        prefix=prefix,
        inputs=inputs,
    )

    del vit_teacher
    del net_student
    torch.cuda.empty_cache()  # Clear the CUDA cache
    gc.collect()  # Collect unused objects
    print("CUDA cache cleared and GPU memory released.")
    print("\n")
