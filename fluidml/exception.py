class NoTasksError(Exception):
    """Exception raised when there are no tasks passed"""

    pass


class TaskResultKeyAlreadyExists(Exception):
    """Exception raised when two tasks produce same key-ed result"""

    pass


class TaskResultObjectMissing(Exception):
    """Exception raised when one or more expected input results could not be retrieved from predecessor tasks"""

    pass


class TaskNameError(Exception):
    """Exception raised when an other object with the name of a task exists.
    Task names have to be unique for pickling.
    """

    pass


class GridSearchExpansionError(Exception):
    """Exception raised when Grid Search expansion fails."""

    pass


class CyclicGraphError(Exception):
    """Exception raised when task spec graph contains circular dependencies."""

    pass


class TmuxError(Exception):
    """Exception raised when tmux bash command returns error"""

    pass
