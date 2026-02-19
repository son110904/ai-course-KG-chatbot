# logger.py
import logging
import os
from datetime import datetime


class Logger:
    """
    Centralized logging utility with file and console output.
    Supports configurable log levels via environment variable.
    """

    def __init__(self, namespace='AppLogger', log_dir='logs', log_file='app.log'):
        """
        Initialize logger with namespace and output configuration.
        
        Args:
            namespace: Logger identifier (e.g., 'DocumentProcessor')
            log_dir: Directory to store log files
            log_file: Log file name
        """
        log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        
        self.logger = logging.getLogger(namespace)
        self.logger.setLevel(log_level)

        # Ensure the logs directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Only add handlers if they are not already added
        if not self.logger.hasHandlers():
            # Create a file handler with timestamp
            timestamp = datetime.now().strftime('%Y%m%d')
            log_file_with_timestamp = f"{timestamp}_{log_file}"
            file_handler = logging.FileHandler(
                os.path.join(log_dir, log_file_with_timestamp)
            )
            file_handler.setLevel(log_level)

            # Create a console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)

            # Create a detailed logging format
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add the handlers to the logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        """Return the configured logger instance."""
        return self.logger