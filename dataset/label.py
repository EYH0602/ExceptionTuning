from returns.maybe import Maybe, Some, Nothing

LABELS = [
    "ValueError",
    "NoError",
    "IndexError",
    "UnknownError",
    "AssertionError",
    "TypeError",
    "AttributeError",
    "ModuleNotFoundError",
    "EOFError",
    "ImportError",
    "KeyError",
    "NameError",
    "OSError",
    "SyntaxError",
    "ZeroDivisionError",
    "UnboundLocalError",
    "UnicodeDecodeError",
    "FileNotFoundError",
    "OverflowError",
    "RecursionError",
    "RuntimeError",
    "PermissionError",
]


def to_int_label(err_type: str) -> Maybe[int]:
    return Some(LABELS.index(err_type)) if err_type in LABELS else Nothing
