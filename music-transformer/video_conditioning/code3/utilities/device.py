# For all things related to devices
#### ONLY USE PROVIDED FUNCTIONS, DO NOT USE GLOBAL CONSTANTS ####
import torch

TORCH_CPU_DEVICE = torch.device('cpu')

if (torch.cuda.device_count() > 0):
    TORCH_CUDA_DEVICE = torch.device('cuda')
else:
    print(
        '---- WARNING: CUDA devices not detected. This will cause the model to run very slow! ----',
    )
    print('')
    TORCH_CUDA_DEVICE = None

USE_CUDA = True


# use_cuda
def use_cuda(cuda_bool):
    """Sets whether to use CUDA (if available), or use the CPU (not recommended).

    Args:
        cuda_bool: use cuda
    """
    global USE_CUDA
    USE_CUDA = cuda_bool


# get_device
def get_device():
    """Grabs the default device.

    Default device is CUDA if available and use_cuda is not False, CPU otherwise.

    Returns:
        device: torch device
    """
    if ((not USE_CUDA) or (TORCH_CUDA_DEVICE is None)):
        return TORCH_CPU_DEVICE
    else:
        return TORCH_CUDA_DEVICE


# cuda_device
def cuda_device():
    """Grabs the cuda device (may be None if CUDA is not available).

    Returns:
        device: torch device
    """
    return TORCH_CUDA_DEVICE


# cpu_device
def cpu_device():
    """Grabs the cpu device.

    Returns:
        device: torch device
    """
    return TORCH_CPU_DEVICE
