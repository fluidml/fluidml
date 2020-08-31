import networkx as nx

from typing import Dict


class Workflow:
    def __init__(self,
                 graph: nx.DiGraph):
        self.graph = graph

    @classmethod
    def from_pipeline(cls,
                      pipeline: Dict[str, str]):
        # create graph
        graph = nx.DiGraph()

        task_to_id = {task: id_ for id_, task in enumerate(pipeline)}
        id_to_task = {id_: task for id_, task in enumerate(pipeline)}

        # add nodes
        for task, dependencies in pipeline.items():
            for dep in dependencies:
                graph.add_edge(task_to_id[dep], task_to_id[task])
        return cls(graph=graph)
