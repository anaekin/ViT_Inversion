# mango_search.py
from mango import scheduler, Tuner
import numpy as np
import sys
from mango.domain.distribution import loguniform
from run_hp_search import run

# Define the hyperparameter space as discrete sets (for grid search):
param_space = dict(
    n_iterations=[3000, 5000, 7000],
    warmup_length=[100, 200, 300],
    lr=loguniform(-3, 3),  # 10^-3 to 1
    first_bn_multiplier=[5.0, 10.0, 15.0],
    main_loss_scale=loguniform(-3, 3),  # 10^-3 to 1
    image_prior_scale=loguniform(-5, 3),  # 10^-5 to 10^-2
    r_feature_scale=loguniform(-5, 3),  # 10^-5 to 10^-2
    tv_l1_scale=loguniform(-6, 3),  # 10^-6 to 10^-3
    tv_l2_scale=loguniform(-6, 3),  # 10^-6 to 10^-3
    l2_scale=loguniform(-6, 3),  # 10^-6 to 10^-3
    # extra_prior_scale=loguniform(-6, 3), # We search for tv_l1, tv_l2 and l2 scale
    patch_prior_scale=loguniform(-5, 3),  # 10^-5 to 10^-2
)

early_stopping_patience = 3


def early_stop(results):
    """
    stop if best objective does not improve for n iterations set by early stopping patience.
    results: dict (same keys as dict returned by tuner.minimize/maximize)
    """
    current_best = results["best_objective"]
    patience_window = results["objective_values"][-(early_stopping_patience):]
    print("current_best", current_best)
    print("patience_window", patience_window)
    return max(patience_window) < current_best


# Mango configuration specifying grid search
mango_config = dict(batch_size=1, num_iteration=100, early_stopping=early_stop)


@scheduler.serial
def objective(**hparams):
    """
    Mango's objective function.
    hparams is a dict that will have the keys from param_dict above.
    """
    params_for_run = {
        # Fix other params
        "local_rank": 0,
        "no_cuda": False,
        "fp16": False,
        "jitter": 30,
        "do_flip": True,
        "store_dir": "test",
        "random_label": False,
        "store_best_images": False,
        "verifier_arch": "mobilenet_v2",
        "bs": 32,
        "vit_feature_loss": "l2",
        # Tunable paparms
        "vit_feature_loss": hparams["vit_feature_loss"],
        "bs": hparams["bs"],
        "n_iterations": hparams["n_iterations"],
        "lr": hparams["lr"],
        "warmup_length": hparams["warmup_length"],
        "first_bn_multiplier": hparams["first_bn_multiplier"],
        "main_loss_scale": hparams["main_loss_scale"],
        "image_prior_scale": hparams["image_prior_scale"],
        "r_feature_scale": hparams["r_feature_scale"],
        "tv_l1_scale": hparams["tv_l1_scale"],
        "tv_l2_scale": hparams["tv_l2_scale"],
        "l2_scale": hparams["l2_scale"],
        "extra_prior_scale": 1.0,  # Keeping this fixed as we are searching for tv_l1, tv_l2 and l2 scale
        # "extra_prior_scale": hparams["extra_prior_scale"],
        "patch_prior_scale": hparams["patch_prior_scale"],
    }

    print(f"Running with params: {params_for_run}")

    # Call deep inversion routine
    accuracy = run(params_for_run)
    print(f"Accuracy: {accuracy}")

    return accuracy


tuner = Tuner(param_space, objective, mango_config)
results = tuner.maximize()  # returns best hyperparams and best objective
print("Results:", results)
print("Best hyperparameters:", results["best_params"])
print("Best accuracy:", results["best_objective"])
