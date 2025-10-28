import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import os
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.transforms.functional import to_tensor, normalize

CLASS_LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

def preprocess_image_for_cam(image_path: str, img_size: int) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((img_size, img_size))
    rgb_img = np.float32(image_resized) / 255
    tensor = to_tensor(image_resized)
    tensor = normalize(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    return tensor.unsqueeze(0), rgb_img

def find_target_layer(model: nn.Module, model_name: str) -> list:
    if 'deit_tiny' in model_name or 'vit_tiny' in model_name:
        return [model.blocks[-1]]
    elif 'resnet50' in model_name:
        return [model.layer4[-1]]
    print(f"Warning: Target layer not defined for model {model_name}. CAM might not work.")
    return None

def get_cam_image(model: nn.Module, config: dict, image_path: str, target_class_idx: int, device: torch.device) -> Image.Image:
    model_name = config['MODEL']['NAME']
    img_size = config['DATA']['IMG_SIZE']

    input_tensor, rgb_img = preprocess_image_for_cam(image_path, img_size)
    target_layers = find_target_layer(model, model_name)

    if target_layers is None:
        print(f"Error: Could not find target layer for {model_name}.")
        return Image.new('RGB', (img_size, img_size), color='red')

    targets_for_cam = [ClassifierOutputTarget(target_class_idx)]

    if 'deit' in model_name or 'vit' in model_name:
        grid_size = img_size // model.patch_embed.patch_size[0]
        cam_algorithm = GradCAMPlusPlus
        reshape_transform = lambda x: reshape_transform_vit(x, height=grid_size, width=grid_size)
        print("Using GradCAM++ for ViT/DeiT...")
    elif 'resnet' in model_name:
        cam_algorithm = GradCAM
        reshape_transform = None
        print("Using GradCAM for ResNet...")
    else:
        print(f"Warning: Unknown CAM method for {model_name}. Defaulting to GradCAM.")
        cam_algorithm = GradCAM
        reshape_transform = None

    with cam_algorithm(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        try:
            grayscale_cam = cam(input_tensor=input_tensor.to(device), targets=targets_for_cam)
            grayscale_cam = grayscale_cam[0, :]
        except Exception as e:
            print(f"Error computing CAM for {model_name}: {e}")
            return Image.new('RGB', (img_size, img_size), color='red')

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cam_image_pil = Image.fromarray(cam_image)

    return cam_image_pil