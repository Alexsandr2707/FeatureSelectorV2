import logging.config
from pathlib import Path
from datetime import datetime

FILE_NAME = "app.log"
TIME_FILE_NAME = datetime.now().strftime("app_%Y.%m.%d_%H:%M:%S.log")
LOG_DIR = Path("logs")
FILE_PATH = LOG_DIR / Path(FILE_NAME)
TIME_FILE_PATH = LOG_DIR / Path(TIME_FILE_NAME)


def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    with open(FILE_PATH, "w"):
        pass
    logging.config.dictConfig(
        {
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
                    "level": "INFO",
                    "formatter": "rich",
                    "rich_tracebacks": True,
                    "markup": True,
                    "show_time": False,
                    "show_level": True,
                    "show_path": False,
                },
                "console": {  # for simple log, default it is unused
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "console",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "file",
                    "mode": "w",
                    "filename": FILE_PATH,
                    "encoding": "utf-8",
                    "maxBytes": 5 * 1024 * 1024,  # 5 MB
                    "backupCount": 5,
                },
                "time_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
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
    )
