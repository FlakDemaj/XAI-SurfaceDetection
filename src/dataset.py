from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Paths / Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sorted"

BATCH_SIZE = 32
RANDOM_SEED = 42

# Image transforms

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.5],
        std = [0.5])
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])

# Create Datasets

def create_datasets():
    base_dataset = ImageFolder(DATA_DIR)

    targets = np.array(base_dataset.targets)
    indices = np.arange(len(base_dataset))

    # 70% Training / 30% temp.
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=RANDOM_SEED)

    # 15 % validation / 15 % test

    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_SEED
    )

    train_full = ImageFolder(DATA_DIR, transform=train_transform)
    valid_full = ImageFolder(DATA_DIR, transform=eval_transform)
    test_full = ImageFolder(DATA_DIR, transform=eval_transform)

    train_dataset = Subset(train_full, train_idx)
    valid_dataset = Subset(valid_full, valid_idx)
    test_dataset = Subset(test_full, test_idx)

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        base_dataset.classes
    )

# Create DataLoaders

def create_dataloaders():

    (train_dataset,
     val_dataset,
     test_dataset,
     class_names) = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        class_names
    )