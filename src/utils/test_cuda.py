import os
import torch

def test_gpu():
    print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES'))
    print("PyTorch Version:", torch.__version__)
    print("CUDA Version in PyTorch:", torch.version.cuda)
    
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    return device

if __name__ == "__main__":
    test_gpu()