import traceback
import torch

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def unit_bounds(d):

    bounds = torch.tensor([[i for _ in range(d)] for i in range(2)],
                          dtype=dtype,
                          device=device)

    return bounds

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    
    print("\n")
    print(f"{category.__name__}: \n {message} \n")
    traceback.print_stack()
    print("\n")

def gpu_warmup():
    device = torch.device("cuda")
    print(f"Warming up GPU: {device}")

    for _ in range(100):
        x = torch.randn(5000, 5000, device=device)
        y = torch.matmul(x, x)
        torch.cuda.synchronize() 

    print("GPU warmed up and ready.")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
