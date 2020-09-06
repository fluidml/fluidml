from busy_bee import Swarm, Task
from typing import List, Any, Dict


class Flow:
    def __init__(self, swarm: Swarm):
        self._swarm = swarm

    def run(self, tasks: List[Task]) -> Dict[str, Dict[str, Any]]:
        """
        Runs the provided tasks

        Args:
            tasks (List[Task]): tasks to run

        Returns:
            Dict[str, Dict[str, any]] - a nested dictionary of results

        """

        # 1. first expand the tasks that are grid searcheable

        # 2. also, take care of their dependencies

        # 3. get a final list of tasks

        # 4. run the tasks through swarm

        # 5. return results
