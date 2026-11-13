"""Domain errors with JSON-Pointer style field paths and zero-based indices."""


class InputError(Exception):
    """Raised when the input document does not satisfy the specification.

    The message is always prefixed with a field path such as ``$.k[0]``.
    """

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")
