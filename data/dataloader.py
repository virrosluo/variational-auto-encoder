import torchvision
import torchvision.transforms as transforms

import torch

from data.config import DataConfig

def get_cifar10_loader(data_config: DataConfig):
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_config.data_dir,
        train=True,
        download=True,
        transform=transform_pipeline
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=data_config.train_batch_size,
        shuffle=True,
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_config.data_dir,
        train=False,
        download=True,
        transform=transform_pipeline
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=data_config.test_batch_size,
        shuffle=False
    )

    return {
        'train': train_loader,
        'test': test_loader
    }

if __name__ == '__main__':
    data_config = DataConfig(
        data_dir='./data_download',
        train_batch_size=256,
        test_batch_size=512
    )

    get_cifar10_loader(data_config)