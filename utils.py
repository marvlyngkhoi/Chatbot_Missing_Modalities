import random
import numpy as np
import torch
from transformers import set_seed as hf_set_seed

def set_seed(seed: int = 42):
    """Set random seed for reproducibility across different libraries."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed_all(seed)  # Ensures all GPUs get the same seed
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior in PyTorch
    torch.backends.cudnn.benchmark = False  # Disables auto-optimization for reproducibility
    #tf.random.set_seed(seed)  # TensorFlow
    hf_set_seed(seed)  # Hugging Face Transformers


