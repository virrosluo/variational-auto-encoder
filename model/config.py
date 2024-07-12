from dataclasses import dataclass, field

@dataclass
class ModelConfig:
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