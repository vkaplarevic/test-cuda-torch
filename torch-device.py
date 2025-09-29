#!/usr/bin/env python
import torch


def get_torch_cuda_version():
    if torch.cuda.is_available():
        print("PyTorch built with CUDA version:", torch.version.cuda)
        print("CUDA device capability:", torch.cuda.get_device_capability())
        print("CUDA device name:", torch.cuda.get_device_name())
    else:
        print("CUDA is not available.")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

get_torch_cuda_version()


print("* * * * * * * * * * * * * ")
print("PyTorch decided to use following device:", device)
print("* * * * * * * * * * * * * ")
