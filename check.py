import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available! You can use GPU acceleration.")
else:
    print("CUDA is not available. You can only use CPU for computation.")