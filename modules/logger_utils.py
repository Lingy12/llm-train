import logging

def get_logger(name):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    # Set up logging
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger(__name__)
    return logger