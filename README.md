# GPU process pool executor

## background
when doing some parallel workload on GPUs, it is very annoying to manage which process to use which GPU.
but using distributed framework like `accelerator`, `torch.distributed` may let code too heavy.

This is a code just modify some code of `concurrent.futures.ProcessPoolExecutor` to enable auto schedue of GPU use, simple interface with origin ProcessPoolExecutor.
This code is suitable for:
- limited on one machine with multi GPUs
- do some eval tasks in machine learning(given a list of text or images, using AI model to process them)
- simple modify of others codebase to enable parallel with minimal change

This code is limited with:
- only support NVIDIA GPUs
- tensors on device could not trans as results

## usage
first copy the `gpu_process.py` to your code base.

```python
from gpu_process import GPUProcessPoolExecutor
gpu_indexs = [0, 1, 2]  # manul select gpu index
gpu_indexs = None # for auto detect gpu via nvidia-smi
with GPUProcessPoolExecutor(gpu_indexs=gpu_indexs) as executor:
    executor.submit(<some functions>, <args>)
```

see `example.py` for more details.

