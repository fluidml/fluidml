class NoTasksError(Exception):
    """Exception raised when there are no tasks passed"""
    pass


class TaskResultKeyAlreadyExists(Exception):
    """Exception raised when two tasks produce same key-ed result"""
    pass
