

import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import _base
from concurrent.futures.process import _ExceptionWithTraceback, _sendback_result
import logging

import test
logging.basicConfig(level=logging.INFO)
import threading
import multiprocessing as mp

def gpu_process_worker(call_queue, result_queue, initializer, initargs, gpu_initializer, gpu_index):
    """Evaluates calls from call_queue and places the results in result_queue.

    This worker is run in a separate process.

    Args:
        call_queue: A ctx.Queue of _CallItems that will be read and
            evaluated by the worker.
        result_queue: A ctx.Queue of _ResultItems that will written
            to by the worker.
        initializer: A callable initializer, or None
        initargs: A tuple of args for the initializer
    """
    # set gpu environment
    if gpu_initializer is None:
        print('gpu_initializer is None, no gpu env will be changed for each child process')
    else:
        try:
            gpu_initializer(gpu_index)
        except BaseException:
            _base.LOGGER.critical('Exception in gpu initializer:', exc_info=True)
    
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException:
            _base.LOGGER.critical('Exception in initializer:', exc_info=True)
            # The parent will notice that the process stopped and
            # mark the pool broken
            return
    while True:
        call_item = call_queue.get(block=True)
        if call_item is None:
            # Wake up queue management thread
            result_queue.put(os.getpid())
            return
        try:
            r = call_item.fn(*call_item.args, **call_item.kwargs)
        except BaseException as e:
            exc = _ExceptionWithTraceback(e, e.__traceback__)
            _sendback_result(result_queue, call_item.work_id, exception=exc)
        else:
            _sendback_result(result_queue, call_item.work_id, result=r)
            del r

        # Liberate the resource as soon as possible, to avoid holding onto
        # open files or shared memory that is not needed anymore
        del call_item

def torch_gpu_set_func(gpu_index):
    import torch
    torch.cuda.set_device(gpu_index)
    
def torch_gpu_list_func():
    import torch
    return list(range(torch.cuda.device_count()))

def env_gpu_set_func(gpu_index):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

def env_gpu_list_func():
    return list(range(int(os.popen("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l").read().strip())))


class GPUProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(self, gpu_indices=None, mp_context=None,
                 initializer=None, initargs=(), 
                 gpu_initializer=torch_gpu_set_func,
                 gpu_list_func=torch_gpu_list_func):
        if gpu_indices is None:
            gpu_indices = gpu_list_func()
        self.gpu_indices = gpu_indices
        self.gpu_initializer = gpu_initializer
        self.gpu_status = [{"using": False, "lock": threading.Lock()} for _ in gpu_indices]
        mp_context = mp.get_context('spawn')
        super().__init__(len(gpu_indices), mp_context, initializer, initargs)
    
    
    def _spawn_process(self):
        for gpu_index, status in zip(self.gpu_indices, self.gpu_status):
            if not status["using"]:
                with status["lock"]:
                    status["using"] = True
                    p = self._mp_context.Process(
                        target=gpu_process_worker,
                        args=(self._call_queue,
                            self._result_queue,
                            self._initializer,
                            self._initargs,
                            self.gpu_initializer,
                            gpu_index))
                    p.start()
                    self._processes[p.pid] = p
                    break




def test_env(n):
    import time
    import random
    import torch
    print(f'task={n}   {torch.cuda.current_device()=}')
    time.sleep(random.random() + 2)
    return True

def test_gpu_tensor_return(n):
    import torch
    device = 'cuda'
    tensor = torch.ones((10,), device=device) * n
    return tensor

def test_gpu_tensor_input(tensor):
    device = 'cuda'
    print('receive', tensor)
    tensor = tensor.to(device)
    return tensor

def test_gpu_tensor_input_rtcpu(tensor):
    print('receive', tensor)
    tensor = tensor.cpu()
    return tensor
    

def test1():
    # basic test
    futures = []
    with GPUProcessPoolExecutor(gpu_indices=[0, 1, 2]) as executor:
        for number in range(12):
            futures.append(executor.submit(test_env, number))
        for f in futures:
            print(f.result())

def test2():
    # test main recevice gpu tensor from different child process, these tensor was allocated by child
    futures = []
    with GPUProcessPoolExecutor(gpu_indices=[0, 1, 2]) as executor:
        for number in range(12):
            futures.append(executor.submit(test_gpu_tensor_return, number))
        for f in futures:
            print(f.result())

def test3():
    # send cpu tensor receive gpu tensor of different gpu device
    futures = []
    tensor_ls = []
    import torch
    with GPUProcessPoolExecutor(gpu_indices=[0, 1, 2]) as executor:
        for number in range(6):
            tensor = torch.ones((10,)) * number
            futures.append(executor.submit(test_gpu_tensor_input, tensor))
        for f in futures:
            print(f.result())
            tensor_ls.append(f.result())
    for t in tensor_ls:
        print(t.mean())

def test4():
    # !!! failed
    # RuntimeError: Attempted to send CUDA tensor received from another process; this is not currently supported. Consider cloning before sending.
    # send gpu tensor on same device receive gpu tensor of different gpu device
    futures = []
    tensor_ls = []
    import torch
    with GPUProcessPoolExecutor(gpu_indices=[0, 1, 2]) as executor:
        for number in range(6):
            tensor = torch.ones((10,), device='cuda') * number
            futures.append(executor.submit(test_gpu_tensor_input, tensor))
        for f in futures:
            print(f.result())
            tensor_ls.append(f.result())
    for t in tensor_ls:
        print(t.mean())


def test5():
    # send gpu tensor on same device, receive cpu tensor
    futures = []
    tensor_ls = []
    import torch
    with GPUProcessPoolExecutor(gpu_indices=[0, 1, 2]) as executor:
        for number in range(6):
            tensor = torch.ones((10,), device='cuda') * number
            futures.append(executor.submit(test_gpu_tensor_input_rtcpu, tensor))
        for f in futures:
            print(f.result())
            tensor_ls.append(f.result())
    for t in tensor_ls:
        print(t.mean())


if __name__ == "__main__":
    
    test1()
    test2()
    test3()
    try:
        test4()
    except Exception as e:
        print(e)
    test5()
    