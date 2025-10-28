import torch
import torch.nn as nn
import os
import yaml
import sys

project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.models.model_factory import create_model
except ImportError:
    print("ERROR: Could not import 'src.models.model_factory'.")
    print("Ensure src/__init__.py exists and the script is run from the project root.")
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


def export_model_to_onnx():
    print("Starting ONNX export process...")

    CONFIG_FILE = os.path.join(project_root, 'config', 'deit_tiny.yaml')
    if not os.path.exists(CONFIG_FILE):
        print(f"ERROR: Configuration file not found at {CONFIG_FILE}")
        return

    config = load_config(CONFIG_FILE)
    RUN_NAME = config.get('RUN_NAME', 'deit_tiny_run') # Use get with default
    CHECKPOINT_NAME = f"{RUN_NAME}_best.pth"
    CHECKPOINT_PATH = os.path.join(project_root, 'models', 'checkpoints', CHECKPOINT_NAME)

    DEVICE = torch.device('cpu')
    print(f"Using device: {DEVICE}")

    model = create_model(config, DEVICE)
    if model is None:
         print("ERROR: Failed to create model.")
         return


    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}.")
        print("Please train the DeiT-Tiny model first.")
        return

    print(f"Loading weights from: {CHECKPOINT_PATH}")
    try:
        # Load checkpoint with weights_only=False for compatibility, but True is safer
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return


    img_size = config.get('DATA', {}).get('IMG_SIZE', 224) # Use get with defaults
    try:
        dummy_input = torch.randn(1, 3, img_size, img_size, device=DEVICE)
    except Exception as e:
        print(f"Error creating dummy input: {e}")
        return


    output_dir = os.path.join(project_root, 'models', 'final')
    os.makedirs(output_dir, exist_ok=True)
    ONNX_OUTPUT_PATH = os.path.join(output_dir, f"{RUN_NAME}.onnx")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_OUTPUT_PATH,
            export_params=True,
            opset_version=14, # Updated opset version
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input' : {0 : 'batch_size'},
                          'output' : {0 : 'batch_size'}}
        )
        print("\n--- ONNX EXPORT SUCCESSFUL! ---")
        print(f"Model saved at: {ONNX_OUTPUT_PATH}")

    except Exception as e:
        print(f"\n--- ERROR DURING ONNX EXPORT ---")
        print(e)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    export_model_to_onnx()