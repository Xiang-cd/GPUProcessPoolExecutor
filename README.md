# GPU process pool executor

## Background
when doing some parallel workload on GPUs, it is very annoying to manage which process to use which GPU.
but using distributed framework like `accelerator`, `torch.distributed` may let code too heavy.

This is a code just modify some code of `concurrent.futures.ProcessPoolExecutor` to enable auto schedue of GPU use, simple interface with origin ProcessPoolExecutor.
This code is suitable for:
- limited on one machine with multi GPUs
- do some eval tasks in machine learning(given a list of text or images, using AI model to process them)
- simple modify of others codebase to enable parallel with minimal change

This code is limited with:
- only support NVIDIA GPUs
- only support torch set device
- tensors on device could not trans as results

## Usage
first copy the `gpu_process.py` to your code base.

```python
from gpu_process import GPUProcessPoolExecutor
gpu_indexs = [0, 1, 2]  # manul select gpu index
gpu_indexs = None # for auto detect gpu via nvidia-smi
with GPUProcessPoolExecutor(gpu_indexs=gpu_indexs) as executor:
    executor.submit(<some functions>, <args>)

# if GPU workload is small and memory is enough
# we could assign two or more workers on a gpu
# this will make gpu more utilized
gpu_indexs = [0, 1, 2, 3] + [0, 1, 2, 3]  
with GPUProcessPoolExecutor(gpu_indexs=gpu_indexs) as executor:
    executor.submit(<some functions>, <args>)
```

see `example.py` for more details.

## How it works and modify
When init each process worker, we could set environment as following to set default device:
```python
    # set gpu environment
    import torch
    torch.cuda.set_device(gpu_index)
```

You could easily modify the code with following:
```python
    # set gpu environment
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
```
acutally when import torch before CUDA_VISIBLE_DEVICES is set, the CUDA_VISIBLE_DEVICES does not work.
So I finally choose to only set device using torch interface.
