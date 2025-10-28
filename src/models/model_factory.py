import timm
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

def create_model(config: dict, device: torch.device):
    model_name = config['MODEL']['NAME']
    pretrained = config['MODEL']['PRETRAINED']
    num_classes = config['NUM_CLASSES']
    
    print(f"Initializing model: {model_name} (Pre-trained: {pretrained})")

    try:
        model = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=num_classes,
        )
        
    except RuntimeError as e:
        print(f"MODEL LOADING ERROR: {e}")
        print("Ensure the model name and number of classes are valid.")
        return None
        
    model.to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            
    return model
    
def create_optimizer_and_scheduler(config: dict, model: nn.Module, len_dataloader: int):
    lr = config['TRAIN']['LEARNING_RATE']
    weight_decay = config['TRAIN']['WEIGHT_DECAY']
    
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    
    if config['TRAIN']['OPTIMIZER_NAME'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Optimizer {config['TRAIN']['OPTIMIZER_NAME']} is not supported.")

    scheduler = None
    if config['TRAIN']['USE_SCHEDULER']:
        total_epochs = config['TRAIN']['EPOCHS']
        warmup_epochs = config['TRAIN']['WARMUP_EPOCHS']
        
        main_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_epochs - warmup_epochs
        )
        
        total_warmup_steps = warmup_epochs * len_dataloader 
        
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: (step / total_warmup_steps) if step < total_warmup_steps else 1.0
        )
        
        scheduler = main_scheduler

    return optimizer, scheduler