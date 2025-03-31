# data/dataloader.py
from torch.utils.data import DataLoader, random_split, Subset
from data.dataset import EmotionDataset
# from data.dataset import collate_emotion_data # Import if using custom collate
import hyperparameters as hp
import numpy as np

def create_dataloaders(dataset, batch_size=hp.BATCH_SIZE, validation_split=hp.VALIDATION_SPLIT, num_workers=hp.NUM_WORKERS, seed=42):
    """Creates training and validation DataLoader objects."""

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    # Ensure reproducibility
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    # Create subsets for train and validation
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Dataset size: {dataset_size}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Create DataLoaders
    # collate_fn = collate_emotion_data # Use if needed
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if hp.DEVICE == 'cuda' else False,
        # collate_fn=collate_fn # Use if needed
        drop_last=True # Good practice if batch sizes vary a lot due to filtering bad data
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, # Can use larger batch size for validation if memory allows
        shuffle=False, # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True if hp.DEVICE == 'cuda' else False,
        # collate_fn=collate_fn # Use if needed
        drop_last=False
    )

    return train_loader, val_loader