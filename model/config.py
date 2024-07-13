from dataclasses import dataclass, field
from typing import Any

@dataclass
class ModelConfig:
    use_reparameterization_trick: bool = field(
        default=True,
        metadata={
            'help': 'Choosing the VAE sampling method for smooth gradient descent (Reparameterization Trick) or not'
        }
    )
    
    encoder_output_dim: int = field(
        default=512,
        metadata={
            'help': 'Output dimensions of `mu` and `log_var`'
        }
    )

    latent_dim: int = field(
        default=256,
        metadata={
            'help': 'Output dimensions of the resnet18 decoder'
        }
    )

@dataclass
class PredictionOutput:
    elbo: Any
    kl_loss: Any
    recon_loss: Any
    input_image: Any
    reconstruct_image: Any