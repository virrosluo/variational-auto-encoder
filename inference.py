import lightning as pl
import torch

from model.lightning_model import vae_lightning

model = vae_lightning.load_from_checkpoint(
    "training_log/lightning_logs/version_0", 
    map_location={"cuda:0", "cpu"}
)