from data.dataloader import get_cifar10_loader

from model import (
    vae_model,
    lightning_model
)

import lightning as pl

from transformers import HfArgumentParser
import sys

from model.config import ModelConfig
from data.config import DataConfig
from config import TrainingArguments

def main():
    parser = HfArgumentParser([ModelConfig, DataConfig, TrainingArguments])
    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        model_args, data_args, train_args = parser.parse_json_file(sys.argv[1])
    else:
        model_args, data_args, train_args = parser.parse_args_into_dataclasses()

# ------------------------------------------------------ DATALOADER PREPARATION
    dataLoader = get_cifar10_loader(data_args)

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

    training_model = lightning_model.vae_lightning(model)

# ------------------------------------------------------ TRAINER PREPARATION
    trainer = pl.Trainer(
        accelerator="auto",
        default_root_dir=train_args.default_root_dir,
        logger=train_args.logger,
        max_epochs=train_args.max_epochs,
        max_steps=train_args.max_steps,
        check_val_every_n_epoch=train_args.check_val_every_n_epoch,
        val_check_interval=train_args.val_check_interval
    )

# ------------------------------------------------------ START TRAINING
    print("[+] Test Metrics Before Training:")
    trainer.test(
        model=training_model,
        dataloaders=dataLoader['test']
    )

    print("[+] START TRAINING")
    trainer.fit(
        model=training_model,
        train_dataloaders=dataLoader['train'],
        val_dataloaders=dataLoader['test']
    )

    print("[+] Test Metrics After Training:")
    trainer.test(
        model=training_model,
        dataloaders=dataLoader['test']
    )

if __name__ == '__main__':
    main()