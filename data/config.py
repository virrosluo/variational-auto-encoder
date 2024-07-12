from dataclasses import dataclass, field

@dataclass
class DataConfig:
    data_dir: str = field(
        default='.',
        metadata={
            'help': 'Directory for storing download dataset'
        }
    )

    train_batch_size: int = field(
        default=256
    )

    test_batch_size: int = field(
        default=512
    )

