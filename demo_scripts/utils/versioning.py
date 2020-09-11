import json
import os
from typing import List, Dict, Optional


def scan_task_dir(task_dir: str) -> List[str]:
    old_run_dirs = [os.path.join(task_dir, d.name)
                    for d in os.scandir(task_dir)
                    if d.is_dir and d.name.isdigit()]
    return old_run_dirs


def make_run_dir(task_dir: str):
    exist_run_dirs = scan_task_dir(task_dir=task_dir)

    new_id = max([int(d) for d in exist_run_dirs]) + 1 if exist_run_dirs else 0
    new_run_dir = os.path.join(task_dir, f'{str(new_id).zfill(3)}')
    os.makedirs(new_run_dir, exist_ok=True)


def task_executed(exist_run_dirs: List[str],
                  task_config: Dict) -> Optional[str]:
    for exist_run_dir in exist_run_dirs:
        try:
            exist_config = json.load(open(os.path.join(exist_run_dir, 'config.json'), 'r'))
        except FileNotFoundError:
            continue

        if task_config == exist_config:
            return exist_run_dir
    return None
