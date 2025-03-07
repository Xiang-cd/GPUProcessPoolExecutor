import os
import torch
from diffusers import CogVideoXPipeline
from gpu_process import GPUProcessPoolExecutor

def gen_one_video(pipe: CogVideoXPipeline, prompt):
    pipe.to('cuda')
    res = pipe(prompt, num_frames=17)
    return res
    

def main():
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    prompt_list = [
        "a man",
        "a woman",
        "rainig",
        "rainbow",
        "rainbow"
    ]
    
    with GPUProcessPoolExecutor([4, 7]) as executor:
        res_ls = []
        for prompt in prompt_list:
            future = executor.submit(gen_one_video, pipe, prompt)
            res_ls.append(future)
        for future in res_ls:
            print(future.result())


if __name__ == "__main__":
    main()