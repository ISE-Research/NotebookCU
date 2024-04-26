import logging
import logging.config

logger_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(module)s lineno=%(lineno)d: %(message)s",
        },
    },
    "handlers": {
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "../logs/app.log",
            "maxBytes": 100_000_000,
            "backupCount": 10,
            "formatter": "standard",
            "level": "INFO",
        },
        "stream_handler": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "DEBUG",
        },
    },
    "loggers": {
        "": {
            "handlers": ["file_handler", "stream_handler"],
            "level": "DEBUG",
            "propagate": False,
        }
    },
}


def init_logger():
    logging.config.dictConfig(logger_config)
