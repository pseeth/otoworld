#!/usr/bin/env python

"""
Script to wait for an open GPU, then to run the job. Use with 
`tsp` (task-spooler) for a common workflow, by queuing a job 
that looks for a GPU and then runs the actual job.

Make this executable: chmod +x allocate.py

This stacks jobs on a single GPU if memory is available and the 
other process belongs to you. Otherwise, it finds a completely
unused GPU to run your command on.

Usage:

# allocates 1 gpu for train script
./allocate.py 1 python train.py ...
# allocates 2 gpus for train script
./allocate.py 2 python train.py ...

Requirements:

pip install nvgpu
"""

import subprocess
import argparse
import time
import logging
import sys
import os, pwd
import nvgpu
from nvgpu.list_gpus import device_statuses

logging.basicConfig(level=logging.INFO)
mem_threshold = 50

def run(cmd):
    print(cmd)
    subprocess.run([cmd], shell=True)

def _allocate_gpu(num_gpus):
    current_user = pwd.getpwuid(os.getuid()).pw_name
    gpu_info = nvgpu.gpu_info()
    device_info = device_statuses()

    # assume nothing is available
    completely_available = [False for _ in gpu_info]
    same_user_available = [False for _ in gpu_info]

    for i, (_info, _device) in enumerate(zip(gpu_info, device_info)):
        completely_available[i] = _device['is_available']
        if _info['mem_used_percent'] < mem_threshold and current_user in _device['users']:
            same_user_available[i] = True

    available_gpus = same_user_available
    if sum(same_user_available) == 0:
        available_gpus = completely_available

    available_gpus = [i for i, val in enumerate(available_gpus) if val]

    return available_gpus[:num_gpus]

if __name__ == "__main__":
    args = sys.argv

    num_gpus = int(sys.argv[1])
    cmd = sys.argv[2:]

    available_gpus = _allocate_gpu(num_gpus)

    while len(available_gpus) < num_gpus:
        logging.info("Waiting for available GPUs. Checking again in 30 seconds.")
        available_gpus = _allocate_gpu(num_gpus)
        time.sleep(30)

    available_gpus = ','.join(map(str, available_gpus))
    CUDA_VISIBLE_DEVICES = f'CUDA_VISIBLE_DEVICES={available_gpus}'
    cmd = ' '.join(cmd)
    cmd = f"{CUDA_VISIBLE_DEVICES} {cmd}"
    run(cmd)
