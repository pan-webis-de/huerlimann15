import logging


def setup_logging(level):
    """
    :param level: logging.INFO or logging.DEBUG, etc.
    """
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    return logging.getLogger(__name__)

# get logger
log = setup_logging(logging.DEBUG)
