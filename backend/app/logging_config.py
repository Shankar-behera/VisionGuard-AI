"""JSON-friendly logging configuration.

Using structured logs makes this deployable behind log aggregators
(Datadog, CloudWatch, Render's log stream, etc.) without extra setup.
"""
import logging
import sys


def configure_logging(log_level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level.upper())
    root.handlers = [handler]

    # Quiet noisy third-party loggers a bit
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
