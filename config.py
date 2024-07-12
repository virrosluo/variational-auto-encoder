from dataclasses import dataclass, field
from typing import (
    Any
)

@dataclass
class TrainingArguments:
    default_root_dir: str = field(
        default='.',
        metadata={
            'help': 'Root dir for saving the training process'
        }
    )

    logger: bool = field(
        default=True,
        metadata={
            'help': 'Does we log the training process'
        }
    )

    max_epochs: int = field(
        default=10,
        metadata={
            'help': 'If `None`, it will use steps size. If both are default -> max_epochs=1000'
        }
    )
    max_steps: int = field(
        default=-1,
        metadata={
            'help': 'If `-1`, it will use epochs size. If both are default -> max_epochs=1000'
        }
    )

    check_val_every_n_epoch: int = field(
        default=1,
        metadata={
            'help': 'Running validation step after n epochs. If `None` it will check after n batch'
        }
    )
    
    val_check_interval: Any = field(
        default=1.0,
        metadata={
            'help': 'Running the validation step after n steps.'
        }
    )