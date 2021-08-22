class FileTypeNotSupported(Exception):
    """Exception raised when files are supplied by the users of a filetype not supported.

    Attributes:
        extension -- extension of the file not supported
        message -- explanation of the error
    """

    def __init__(
        self,
        extension,
        message="Extension: {} not supported. Please supply either csv or npz files.\n",
    ):
        self.extension = extension
        self.message = message.format(extension)
        super().__init__(self.message)

    def __str__(self):
        return self.message


class InvalidDataDefinitionException(Exception):
    """Exception raised when files are supplied by the users of a filetype not supported.

    Attributes:
        extension -- extension of the file not supported
        message -- explanation of the error
    """

    def __init__(
        self,
        extension,
        message="Invalid data defition. Please make sure your data definition adheres to the design outlaid in the divina documentation.\n",
    ):
        self.extension = extension
        self.message = message.format(extension)
        super().__init__(self.message)

    def __str__(self):
        return self.message
