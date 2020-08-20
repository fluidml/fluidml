from rich import console


class Console:
    __instance = None
    @staticmethod 
    def get_instance():
        """ Static access method. """
        if Console.__instance is None:
            Console()
        return Console.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Console.__instance is not None:
            raise Exception("Use Console.get_instance()!")
        else:
            Console.__instance = console.Console()