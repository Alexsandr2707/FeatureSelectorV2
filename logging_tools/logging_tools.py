import logging
from functools import wraps
from rich.pretty import pretty_repr
from typing import Any
import textwrap

logger = logging.getLogger(__name__)

CLASS_FORMAT = "%s"
FUNC_FORMAT = "%s"
METHOD_FORMAT = CLASS_FORMAT + "." + FUNC_FORMAT


def log_method(level=logging.DEBUG):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            name = self.__class__.__name__
            logger.log(level, "=" * 50)
            logger.log(
                level,
                CLASS_FORMAT + ": START " + METHOD_FORMAT,
                name,
                name,
                func.__name__,
            )
            result = func(self, *args, **kwargs)
            logger.log(
                level,
                CLASS_FORMAT + ": END " + METHOD_FORMAT,
                name,
                name,
                func.__name__,
            )
            logger.log(level, "=" * 50)

            return result

        return wrapper

    return decorator


def make_obj_logger(self):
    def log(msg: str, *args, level: int = logging.DEBUG, **kwargs):
        msg = CLASS_FORMAT + ": " + msg
        logger.log(level, msg, self.__class__.__name__, *args, **kwargs)

    return log


def make_obj_params_logger(self):
    log = make_obj_logger(self)

    def log_params(msg: str, *params: Any, level=logging.DEBUG):
        params = tuple(map(lambda p: pretty_repr(p), params))
        format = "\n" + "\n".join(["%s"] * len(params))
        ident = " " * 4
        text = textwrap.indent(format % params, ident)
        log(msg + text, level=level)

    return log_params


class ClassLogger:
    def __init__(self):
        self.log = make_obj_logger(self)
        self.log_params = make_obj_params_logger(self)
