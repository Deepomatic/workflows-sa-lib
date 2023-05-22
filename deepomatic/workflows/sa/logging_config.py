import logging

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)

# To switch if process/thread info nessary
FORMAT = "[%(levelname)s %(name)s %(filename)s:%(lineno)s] %(message)s"
formatter = logging.Formatter(fmt=FORMAT)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
