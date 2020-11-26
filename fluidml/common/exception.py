class NoTasksError(Exception):
    """Exception raised when there are no tasks passed"""
    pass


class TaskResultTypeError(Exception):
    """Exception raised when task result is not a dictionary"""
    pass
