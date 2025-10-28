import torchvision.transforms as T
from timm.data import create_transform
import torch

def get_base_transforms(img_size: int, split: str):
    scale_size = int(img_size / 0.875)
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    transforms_list = [
        T.Resize(scale_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ]
    return T.Compose(transforms_list)

def get_strong_augmentation(img_size: int, config: dict):
    ra_mag = config['AUGMENTATION']['RAND_AUGMENT_MAGNITUDE']
    ra_num = config['AUGMENTATION']['RAND_AUGMENT_NUM_OPS']
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    transform = create_transform(
        input_size=img_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m{}-n{}'.format(ra_mag, ra_num),
        interpolation='bicubic',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.25,
        hflip=0.5,
        vflip=0.5,
    )
    return transform

def get_transforms(config: dict, split: str):
    img_size = config['DATA']['IMG_SIZE']
    if split in ['val', 'test']:
        return get_base_transforms(img_size, split)
    elif split == 'train':
        return {
            'base': get_base_transforms(img_size, split),
            'strong': get_strong_augmentation(img_size, config)
        }
    else:
        raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")