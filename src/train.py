import torch
import torch.nn as nn
import os
import yaml
import json
import sys
import argparse
import numpy as np # Needed for serializing numpy types in history
import traceback

project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.models.model_factory import create_model, create_optimizer_and_scheduler
    from src.data.skin_dataset import get_dataloaders
    from src.training.trainer import Trainer
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure you are running this script from the project root directory")
    print("and that src/__init__.py exists.")
    sys.exit(1)


def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)

def main(config_path: str):
    config = load_config(config_path)
    print("Loaded configuration from:", config_path)

    requested_device = config.get('DEVICE', 'cpu')
    if requested_device == 'cuda' and torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device('cpu')
        if requested_device == 'cuda':
             print("CUDA requested but not available. Using CPU.")
        else:
             print("Using CPU.")

    seed = config.get('SEED', 42)
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    print(f"Set random seed to: {seed}")


    try:
        train_loader, val_loader, test_loader = get_dataloaders(config)
        if train_loader is None or val_loader is None:
             print("ERROR: Failed to create train or validation data loaders. Check dataset paths and integrity.")
             return
        print("DataLoaders created successfully.")
    except Exception as e:
        print(f"Error creating DataLoaders: {e}")
        return

    if train_loader is None or len(train_loader) == 0: # Check if train_loader itself is None or empty
         print("ERROR: Training DataLoader is None or empty. Check training data.")
         return
    len_dataloader = len(train_loader)


    model = create_model(config, DEVICE)
    if model is None:
        print("Failed to initialize model. Exiting.")
        return

    try:
        optimizer, scheduler = create_optimizer_and_scheduler(config, model, len_dataloader)
        print("Optimizer and Scheduler created successfully.")
    except Exception as e:
         print(f"Error creating optimizer/scheduler: {e}")
         return


    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=DEVICE
        )
        print("Trainer initialized.")
    except Exception as e:
        print(f"Error initializing Trainer: {e}")
        return

    history = {} # Initialize history dictionary
    try:
        best_model, history = trainer.run()
        print("\n--- TRAINING PROCESS COMPLETED ---")
    except Exception as e:
        print(f"\n--- ERROR DURING TRAINING ---")
        print(f"Error: {e}")
        traceback.print_exc()
        if hasattr(trainer, 'history'):
             history = trainer.history


    history_log_dir = os.path.join('results', 'logs')
    os.makedirs(history_log_dir, exist_ok=True)
    history_file_path = os.path.join(history_log_dir, f"{config.get('RUN_NAME', 'run')}_history.json")

    try:
        serializable_history = {}
        for key, value_list in history.items():
             # Convert numpy types to native Python types for JSON serialization
             serializable_history[key] = [
                 item.item() if isinstance(item, (np.generic, torch.Tensor)) else item
                 for item in value_list
            ]

        with open(history_file_path, 'w') as f:
            json.dump(serializable_history, f, indent=4)
        print(f"Training history saved to: {history_file_path}")
    except Exception as e:
        print(f"Error saving history file: {e}")


if __name__ == '__main__':
    DEFAULT_CONFIG = 'config/resnet50_pre.yaml' # Default config file

    parser = argparse.ArgumentParser(description='Train a skin cancer classification model.')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                        help=f'Path to the configuration YAML file (default: {DEFAULT_CONFIG})')
    args = parser.parse_args()

    CONFIG_FILE = args.config

    if not os.path.exists(CONFIG_FILE):
        print(f"ERROR: Configuration file not found at {CONFIG_FILE}. Please ensure the path is correct and run from the project root.")
    else:
        main(CONFIG_FILE)