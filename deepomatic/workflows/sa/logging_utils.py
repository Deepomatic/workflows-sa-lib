import logging

logger = logging.getLogger(__name__)
# To switch if process/thread info nessary
FORMAT = "[%(levelname)s %(name)s %(filename)s:%(lineno)s] %(message)s"
formatter = logging.Formatter(fmt=FORMAT)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
