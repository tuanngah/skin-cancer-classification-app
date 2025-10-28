import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import os
import torch
import traceback

CLASS_LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

ONNX_MODEL_PATH = "models/final/deit-tiny-base-run-focal-gamma2.onnx"

def preprocess(image_pil: Image.Image) -> np.ndarray:
    scale_size = int(IMG_SIZE / 0.875)
    transforms = Compose([
        Resize(scale_size, interpolation=F.InterpolationMode.BICUBIC),
        CenterCrop(IMG_SIZE),
        ToTensor(),
    ])
    tensor = transforms(image_pil)
    tensor = F.normalize(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    input_np = tensor.unsqueeze(0).numpy()
    return input_np

def load_onnx_session(model_path: str = ONNX_MODEL_PATH, use_gpu: bool = False) -> ort.InferenceSession:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"ONNX session loaded successfully using {session.get_providers()}.")
        return session
    except Exception as e:
        print(f"Error loading ONNX session from {model_path}: {e}")
        raise

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def postprocess(output_logits: np.ndarray) -> dict:
    probabilities = softmax(output_logits)[0]
    results = {CLASS_LABELS[i]: float(probabilities[i]) for i in range(len(CLASS_LABELS))}
    return results

class Predictor:
    def __init__(self, use_gpu: bool = False):
        self.session = load_onnx_session(use_gpu=use_gpu)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image_pil: Image.Image) -> dict:
        input_data = preprocess(image_pil)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        output_logits = outputs[0]
        predictions = postprocess(output_logits)
        return predictions

if __name__ == '__main__':
    try:
        dummy_image = Image.new('RGB', (500, 400), color = 'blue')
        print("Created dummy image.")
        use_gpu_test = torch.cuda.is_available()
        print(f"Attempting to initialize Predictor with use_gpu={use_gpu_test}")
        predictor = Predictor(use_gpu=use_gpu_test)
        print("Predictor initialized.")
        print("Running prediction on dummy image...")
        prediction_result = predictor.predict(dummy_image)
        print("Prediction successful!")
        print("Prediction Result:", prediction_result)
        top_pred = max(prediction_result, key=prediction_result.get)
        print(f"Top prediction: {top_pred} ({prediction_result[top_pred]:.2%})")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        traceback.print_exc()