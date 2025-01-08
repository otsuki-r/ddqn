import logging
import os
import sys

logger = logging.getLogger(__name__)
log_level = os.environ.get("LOG_LEVEL", "DEBUG").lower()
logger.setLevel(
    logging.DEBUG
    if log_level.lower() == "debug"
    else logging.INFO
    if log_level.lower() == "info"
    else logging.WARNING
    if log_level.lower() == "warning"
    else logging.ERROR
)
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
