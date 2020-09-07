from typing import List

import torch


def get_balanced_devices(count: int,
                         no_cuda: bool = False) -> List[str]:
    if not no_cuda and torch.cuda.is_available():
        devices = [f'cuda:{id_}' for id_ in range(torch.cuda.device_count())]
    else:
        devices = ['cpu']
    factor = int(count / len(devices))
    remainder = count % len(devices)
    devices = devices * factor + devices[:remainder]
    return devices
