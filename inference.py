import torch
import sys
import lightning as pl
from transformers import HfArgumentParser

from model.config import ModelConfig
from data.config import DataConfig
from config import TrainingArguments

from data.dataloader import (
    get_cifar10_loader,
    cifar10_normalization
)
from data.config import DataConfig

from model import (
    vae_model,
    lightning_model
)

from matplotlib.pyplot import imshow, figure
import numpy as np
from torchvision.utils import make_grid

def main():
    parser = HfArgumentParser([ModelConfig, DataConfig, TrainingArguments])
    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        model_args, data_args, train_args = parser.parse_json_file(sys.argv[1])
    else:
        model_args, data_args, train_args = parser.parse_args_into_dataclasses()

# ------------------------------------------------------ DATALOADER PREPARATION
    data_config = DataConfig(
        data_dir="data_download",
        test_batch_size=16
    )

    dataLoader = get_cifar10_loader(data_config)

# ------------------------------------------------------ MODEL PREPARATION
    if model_args.use_reparameterization_trick:
        model = vae_model.ReparameterizeVAE(
            enc_out_dim=model_args.encoder_output_dim,
            latent_dim=model_args.latent_dim,
        )
    else:
        model = vae_model.NonReparameterizeVAE(
            enc_out_dim=model_args.encoder_output_dim,
            latent_dim=model_args.latent_dim,
        )

    pretrained_model = lightning_model.vae_lightning(model)

# ------------------------------------------------------ TRAINER PREPARATION
    trainer = pl.Trainer(accelerator="auto")

    output = trainer.predict(
        model=pretrained_model,
        dataloaders=dataLoader['test'],
        ckpt_path="training_log/lightning_logs/version_1/checkpoints/epoch=4-step=487.ckpt"
        )

# ------------------------------------------------------ OUTPUT IMAGE
    figure(figsize=(8, 3), dpi=300)
    normalize = cifar10_normalization()
    mean, std = np.array(normalize.mean), np.array(normalize.std)
    origin_imgs = np.clip(
        make_grid(output[0].input_image).permute(1, 2, 0).numpy() * std + mean,
        a_min=0.,
        a_max=1.,
    )

    figure(figsize=(8, 3), dpi=300)
    normalize = cifar10_normalization()
    mean, std = np.array(normalize.mean), np.array(normalize.std)
    reconstruct_imgs = np.clip(
        make_grid(output[0].reconstruct_image).permute(1, 2, 0).numpy() * std + mean,
        a_min=0.,
        a_max=1.,
    )

    imshow(origin_imgs)
    imshow(reconstruct_imgs)

if __name__ == "__main__":
    main()