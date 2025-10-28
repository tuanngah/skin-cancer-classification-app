import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def calculate_metrics(y_true: torch.Tensor, y_pred_logits: torch.Tensor, num_classes: int = 7) -> dict:
    # Ensure tensors are on CPU and handle potential empty tensors
    if y_true.numel() == 0 or y_pred_logits.numel() == 0:
        # Return default zero metrics if input tensors are empty
        default_cm = np.zeros((num_classes, num_classes), dtype=int)
        default_probs = np.array([])
        default_per_class = {f'f1_{label}': 0.0 for label in CLASS_LABELS}
        results = {
            'accuracy': 0.0, 'macro_f1': 0.0, 'auc_macro_ovr': 0.0,
            'confusion_matrix': default_cm, 'y_pred_probs': default_probs
        }
        results.update(default_per_class)
        return results
        
    y_true_np = y_true.cpu().numpy()
    y_pred_np = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    
    accuracy = accuracy_score(y_true_np, y_pred_np)
    macro_f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    
    # Calculate per-class F1, ensuring array has length num_classes
    per_class_f1_array = f1_score(y_true_np, y_pred_np, average=None, labels=np.arange(num_classes), zero_division=0)
    
    per_class_f1 = {}
    for i, label in enumerate(CLASS_LABELS):
        # Check index boundary before accessing the array
        if i < len(per_class_f1_array):
            per_class_f1[f'f1_{label}'] = per_class_f1_array[i]
        else:
            per_class_f1[f'f1_{label}'] = 0.0 # Assign 0 if class index exceeds array length (shouldn't happen with labels=...)


    conf_matrix = confusion_matrix(y_true_np, y_pred_np, labels=np.arange(num_classes))
    
    y_pred_probs = torch.softmax(y_pred_logits, dim=1).cpu().numpy()
    
    auc_score = 0.0 # Default value
    # Check if there are at least two classes present in y_true_np to calculate AUC
    if len(np.unique(y_true_np)) > 1:
        try:
            auc_score = roc_auc_score(
                y_true_np,
                y_pred_probs,
                multi_class='ovr',
                average='macro',
                labels=np.arange(num_classes) # Explicitly provide labels
            )
        except ValueError as e:
            print(f"Warning: Could not calculate AUC score. Error: {e}")
            auc_score = 0.0
    else:
        print("Warning: Only one class present in y_true. AUC score is not defined and set to 0.0.")


    results = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'auc_macro_ovr': auc_score,
        'confusion_matrix': conf_matrix,
        'y_pred_probs': y_pred_probs
    }
    
    results.update(per_class_f1)
    
    return results

def plot_confusion_matrix(cm: np.ndarray, labels: list = CLASS_LABELS, save_path: str = None):
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path)
            print(f"Confusion matrix plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving confusion matrix plot: {e}")
            
    # plt.show() # Typically called from the notebook, not within the function itself