import logging.config
from pathlib import Path
from datetime import datetime
import re
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

FILE_NAME = "app.log"
TIME_FILE_NAME = datetime.now().strftime("app_%Y.%m.%d_%H:%M:%S.log")
LOG_DIR = Path("logs")
FILE_PATH = LOG_DIR / Path(FILE_NAME)
TIME_FILE_PATH = LOG_DIR / Path(TIME_FILE_NAME)
TIME_FILE_PATTERN = re.compile(r"app_(\d{4}\.\d{2}\.\d{2}_\d{2}:\d{2}:\d{2}).*\.log")
TIME_FILE_COUNT = 10


def parse_dt(name: str):
    m = TIME_FILE_PATTERN.fullmatch(name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y.%m.%d_%H:%M:%S")


def keep_last_n_time_files(directory: str | Path, n: int = 10) -> None:
    p = Path(directory)

    parsed_files = []
    for f in p.iterdir():
        if not f.is_file():
            continue
        dt = parse_dt(f.name)
        if dt:
            parsed_files.append((dt, f))

    parsed_files.sort(key=lambda x: x[0], reverse=True)

    for _, f in parsed_files[n:]:
        try:
            f.unlink()
        except Exception as e:
            print(f"Can't remove file '{f}': {e}")


def setup_logging(
    root_only: bool = False,
    console_level: LogLevel = "INFO",
    file_level: LogLevel = "DEBUG",
    time_file_level: LogLevel = "DEBUG",
) -> None:
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    keep_last_n_time_files(LOG_DIR, n=TIME_FILE_COUNT)
    with open(FILE_PATH, "w"):
        pass

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "rich": {"format": "%(message)s"},
            "console": {"format": "[%(levelname)s]: %(message)s"},
            "file": {
                "format": "[%(levelname)s]: %(message)s",
            },
        },
        "handlers": {
            "rich": {  # for rich log, using as default
                "class": "rich.logging.RichHandler",
                "level": console_level,
                "formatter": "rich",
                "rich_tracebacks": True,
                "markup": True,
                "show_time": False,
                "show_level": True,
                "show_path": False,
            },
            "console": {  # for simple log, default it is unused
                "class": "logging.StreamHandler",
                "level": console_level,
                "formatter": "console",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": file_level,
                "formatter": "file",
                "mode": "w",
                "filename": FILE_PATH,
                "encoding": "utf-8",
                "maxBytes": 5 * 1024 * 1024,  # 5 MB
                "backupCount": 5,
            },
            "time_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": time_file_level,
                "formatter": "file",
                "filename": TIME_FILE_PATH,
                "encoding": "utf-8",
                "maxBytes": 5 * 1024 * 1024,  # 5 MB
                "backupCount": 5,
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["file", "rich", "time_file"],
        },
        "loggers": {
            "matplotlib": {"level": "WARNING", "propagate": True},
            "PIL": {"level": "WARNING", "propagate": True},
        },
    }

    if root_only:
        config["loggers"]["method"] = {"level": "WARNING", "propagate": True}

    logging.config.dictConfig(config)
