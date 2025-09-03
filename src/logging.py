import logging


def setup_logging():
    # Configure the basic settings for logging
    logging.basicConfig(
        level=logging.INFO,  # Sets the minimum severity level to log
        format='%(asctime)s - %(levelname)s - %(message)s'  # Defines the format of the log messages
    )