class NoTasksError(Exception):
    """Exception raised when there are no tasks passed"""
    pass


class TaskResultKeyAlreadyExists(Exception):
    """Exception raised when two tasks produce same key-ed result"""
    pass


class TaskResultObjectMissing(Exception):
    """Exception raised when one or more expected input results could not be retrieved from predecessor tasks"""
    pass


class TaskPublishesSpecMissing(Exception):
    """Exception raised when `publishes` specification is missing in both task specification or task definition"""
    pass


class GridSearchExpansionError(Exception):
    """Exception raised when Grid Search expansion fails."""
    pass
