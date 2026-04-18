import logging
from functools import wraps
from rich.pretty import pretty_repr
from typing import Any
import textwrap
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CLASS_FORMAT = "%s"
FUNC_FORMAT = "%s"
METHOD_FORMAT = CLASS_FORMAT + "." + FUNC_FORMAT


@dataclass
class ExecContext:
    run_id: str
    path: str
    depth: str


def log_method(level=logging.DEBUG):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            logger = getattr(self, "logger", None)
            if logger is None:
                logger = logging.getLogger(self.__class__.__module__)

            cls_name = self.__class__.__name__
            method_name = func.__name__

            logger.log(level, METHOD_FORMAT + " start", cls_name, method_name)
            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            duration = time.perf_counter() - start

            logger.log(
                level,
                METHOD_FORMAT + " end (%.2f)s",
                cls_name,
                method_name,
                duration,
            )
            return result

        return wrapper

    return decorator


def make_obj_logger(self):
    def log(msg: str, *args, level: int = logging.DEBUG, **kwargs):
        msg = CLASS_FORMAT + " " + msg
        logger.log(level, msg, self.__class__.__name__, *args, **kwargs)

    return log


def make_obj_params_logger(self):
    log = make_obj_logger(self)

    def log_params(msg: str, *params: Any, one_line=True, level=logging.DEBUG):
        if one_line:
            format = " " + " | ".join(["%s"] * len(params))
            log(msg + format, *params, level=level)
        else:
            format = "\n" + "\n".join(["%s"] * len(params))
            ident = " " * 4
            text = textwrap.indent(format % params, ident)
            log(msg + text, level=level)

    return log_params


class ClassLogger:
    def __init__(self):
        self.log = make_obj_logger(self)
        self.log_params = make_obj_params_logger(self)
