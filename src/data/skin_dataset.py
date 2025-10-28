import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import os
import numpy as np
from collections import Counter
from src.data.transforms import get_transforms

CLASS_LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
LABEL_MAPPING = {label: i for i, label in enumerate(CLASS_LABELS)}

class SkinCancerDataset(Dataset):
    def __init__(self, config: dict, split: str):
        self.config = config
        self.split = split
        project_root = os.getcwd()

        if split == 'train':
            csv_key = 'TRAIN_CSV'
        elif split == 'val':
            csv_key = 'VAL_CSV'
        elif split == 'test':
            csv_key = 'TEST_CSV'
        else:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")

        relative_csv_path = config['DATA'][csv_key]
        csv_path = os.path.join(project_root, relative_csv_path)

        try:
            self.metadata = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"ERROR: Metadata file not found at path: {csv_path}")
            raise

        relative_image_root = config['DATA']['IMAGE_ROOT']
        self.image_root = os.path.join(project_root, relative_image_root)

        self.metadata['label_idx'] = self.metadata['dx'].map(LABEL_MAPPING)
        self.transforms = get_transforms(config, split)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_id = row['image_id']
        label = row['label_idx']
        img_path = os.path.join(self.image_root, f'{img_id}.jpg')

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"ERROR: Image file not found at path: {img_path}")
            # Return a placeholder tensor and an invalid label (-1)
            dummy_tensor = torch.zeros((3, self.config['DATA']['IMG_SIZE'], self.config['DATA']['IMG_SIZE']))
            return dummy_tensor, torch.tensor(-1, dtype=torch.long)

        if self.split == 'train':
            dx = row['dx']
            if dx == 'nv':
                transformed_image = self.transforms['base'](image)
            else:
                transformed_image = self.transforms['strong'](image)
        else:
            transformed_image = self.transforms(image)

        return transformed_image, torch.tensor(label, dtype=torch.long)

def create_weighted_sampler(dataset: SkinCancerDataset):
    # Ensure metadata is not empty and label_idx exists
    if dataset.metadata.empty or 'label_idx' not in dataset.metadata.columns:
        print("Warning: Metadata is empty or 'label_idx' column missing. Cannot create weighted sampler.")
        return None
        
    class_counts = dataset.metadata['label_idx'].value_counts().sort_index().to_dict()
    if not class_counts:
        print("Warning: No class counts found. Cannot create weighted sampler.")
        return None
        
    total_samples = sum(class_counts.values())
    if total_samples == 0:
        print("Warning: Total samples is zero. Cannot create weighted sampler.")
        return None

    # Calculate weights, handle potential division by zero if a class has 0 samples (though unlikely after value_counts)
    class_weights = {cls_idx: total_samples / count for cls_idx, count in class_counts.items() if count > 0}
    
    # Assign weights, defaulting to 0 if a label isn't in class_weights (shouldn't happen with valid data)
    weights = [class_weights.get(label, 0) for label in dataset.metadata['label_idx']]

    if not weights or sum(weights) == 0:
         print("Warning: Calculated weights list is empty or sums to zero. Cannot create weighted sampler.")
         return None

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    return sampler

def get_dataloaders(config: dict):
    try:
        train_dataset = SkinCancerDataset(config, 'train')
        val_dataset = SkinCancerDataset(config, 'val')
        test_dataset = SkinCancerDataset(config, 'test')
    except Exception as e:
        print(f"Error creating datasets: {e}")
        raise

    train_sampler = None
    shuffle_train = True
    if config.get('DATA', {}).get('USE_WEIGHTED_SAMPLER', False) and len(train_dataset) > 0:
        train_sampler = create_weighted_sampler(train_dataset)
        if train_sampler:
            shuffle_train = False
            print("Using WeightedRandomSampler for training.")
        else:
             print("Warning: Could not create weighted sampler, using standard shuffling.")
             shuffle_train = True # Fallback to shuffling if sampler creation failed
    else:
        print("WeightedRandomSampler not enabled or training dataset empty.")


    num_workers = config.get('DATA', {}).get('NUM_WORKERS', 4) # Default to 4
    batch_size = config.get('TRAIN', {}).get('BATCH_SIZE', 32) # Default to 32

    # Handle potential empty datasets for loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True,
        drop_last=True if shuffle_train or train_sampler else False # Drop last incomplete batch if using sampler/shuffle
    ) if len(train_dataset) > 0 else None

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    ) if len(val_dataset) > 0 else None

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    ) if len(test_dataset) > 0 else None
    
    # Check if any loader failed to initialize (due to empty dataset)
    if train_loader is None or val_loader is None or test_loader is None:
        print("Warning: One or more DataLoaders could not be created (likely due to empty datasets).")


    return train_loader, val_loader, test_loader