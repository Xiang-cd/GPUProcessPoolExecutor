# GPU process pool executor

## Background
when doing some parallel workload on GPUs, it is very annoying to manage which process to use which GPU.
but using distributed framework like `accelerator`, `torch.distributed` may let code too heavy.

This is a code just modify some code of `concurrent.futures.ProcessPoolExecutor` to enable auto schedue of GPU use, simple interface with origin ProcessPoolExecutor.
This code is suitable for:
- limited on one machine with multi GPUs
- do some eval tasks in machine learning(given a list of text or images, using AI model to process them)
- simple modify of others codebase to enable parallel with minimal change


## Usage
first copy the `gpu_process.py` to your code base.

```python
from gpu_process import GPUProcessPoolExecutor
gpu_indices = [0, 1, 2]  # manul select gpu index
gpu_indices = None # for auto detect gpu via nvidia-smi
with GPUProcessPoolExecutor(gpu_indices=gpu_indices) as executor:
    executor.submit(<some functions>, <args>)

# if GPU workload is small and memory is enough
# we could assign two or more workers on a gpu
# this will make gpu more utilized
gpu_indices = [0, 1, 2, 3] + [0, 1, 2, 3]  
with GPUProcessPoolExecutor(gpu_indices=gpu_indices) as executor:
    executor.submit(<some functions>, <args>)
```

see `example.py` for more details.

## How it works and modify
1. When init each process worker, we could set environment by set callable `gpu_initializer` functions to set default device.
2. When gpu_indices not provided, a list of int will be get by `gpu_list_func`.
3. both `gpu_initializer` and `gpu_list_func` are default as pytorch gpu interface.

```python
def torch_gpu_set_func(gpu_index):
    import torch
    torch.cuda.set_device(gpu_index)
    
def torch_gpu_list_func():
    import torch
    return list(range(torch.cuda.device_count()))

GPUProcessPoolExecutor(gpu_indices=None, 
                gpu_initializer=torch_gpu_set_func,
                gpu_list_func=torch_gpu_list_func):
```

