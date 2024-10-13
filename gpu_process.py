

import os
import multiprocessing as mp
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import _base


# Hack to embed stringification of remote traceback in local traceback

class _RemoteTraceback(Exception):
    def __init__(self, tb):
        self.tb = tb
    def __str__(self):
        return self.tb

class _ExceptionWithTraceback:
    def __init__(self, exc, tb):
        tb = traceback.format_exception(type(exc), exc, tb)
        tb = ''.join(tb)
        self.exc = exc
        # Traceback object needs to be garbage-collected as its frames
        # contain references to all the objects in the exception scope
        self.exc.__traceback__ = None
        self.tb = '\n"""\n%s"""' % tb
    def __reduce__(self):
        return _rebuild_exc, (self.exc, self.tb)

def _rebuild_exc(exc, tb):
    exc.__cause__ = _RemoteTraceback(tb)
    return exc



class _ResultItem(object):
    def __init__(self, work_id, exception=None, result=None):
        self.work_id = work_id
        self.exception = exception
        self.result = result

class GPU_ResultItem(_ResultItem):
    def __init__(self, work_id, exception=None, result=None, gpu_index=None):
        super().__init__(work_id, exception, result)
        self.gpu_index = gpu_index



def _sendback_result(result_queue, work_id, result=None, exception=None):
    """Safely send back the given result or exception"""
    try:
        result_queue.put(_ResultItem(work_id, result=result,
                                     exception=exception))
    except BaseException as e:
        exc = _ExceptionWithTraceback(e, e.__traceback__)
        result_queue.put(_ResultItem(work_id, exception=exc))



import logging
logging.basicConfig(level=logging.INFO)


def gpu_sendback_result(result_queue, work_id, result=None, exception=None, gpu_index=None):
    """Safely send back the given result or exception"""
    try:
        result_queue.put(GPU_ResultItem(work_id, result=result,
                                     exception=exception, gpu_index=gpu_index))
    except BaseException as e:
        exc = _ExceptionWithTraceback(e, e.__traceback__)
        result_queue.put(GPU_ResultItem(work_id, exception=exc, gpu_index=gpu_index))
    print("sendding back, gpu: ", gpu_index)

def gpu_process_worker(call_queue, result_queue, initializer, initargs, gpu_index):
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
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    torch.cuda.set_device(gpu_index)
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
            gpu_sendback_result(result_queue, call_item.work_id, exception=exc, gpu_index=gpu_index)
        else:
            gpu_sendback_result(result_queue, call_item.work_id, result=r, gpu_index=gpu_index)
            del r

        # Liberate the resource as soon as possible, to avoid holding onto
        # open files or shared memory that is not needed anymore
        del call_item

class GPUProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(self, gpu_indexs=None, mp_context=None,
                 initializer=None, initargs=()):
        if gpu_indexs is None:
            gpu_indexs = list(range(int(os.popen("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l").read().strip())))
        self.gpu_indexs = gpu_indexs
        self.gpu_status = [{"using": False, "lock": threading.Lock()} for _ in gpu_indexs]
        mp_context = mp.get_context('spawn')
        super().__init__(len(gpu_indexs), mp_context, initializer, initargs)
    
    
    def _spawn_process(self):
        for gpu_index, status in zip(self.gpu_indexs, self.gpu_status):
            if not status["using"]:
                with status["lock"]:
                    status["using"] = True
                    p = self._mp_context.Process(
                        target=gpu_process_worker,
                        args=(self._call_queue,
                            self._result_queue,
                            self._initializer,
                            self._initargs,
                            gpu_index))
                    p.start()
                    self._processes[p.pid] = p
                    break




if __name__ == "__main__":
    def test(n):
        import time
        import random
        print(f"{n} cuda visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', None)}")
        time.sleep(random.random())
        return True

    def main():
        import time
        with GPUProcessPoolExecutor(gpu_indexs=[0, 1, 2]) as executor:
            for number in range(6):
                executor.submit(test, number)
            time.sleep(5)
            for number in range(6, 12):
                executor.submit(test, number)
    main()