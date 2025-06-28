"""
Custom logging configuration for PolicyPal project.
Provides structured logging with different levels and outputs.
"""

import sys
from pathlib import Path
from loguru import logger
import yaml
from typing import Optional, Any


class PolicyPalLogger:
    """Centralized logging configuration for PolicyPal."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the logger with configuration.
        
        Args:
            config_path: Path to logging configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logger()
    
    def _load_config(self, config_path: Optional[str]) -> dict[str, Any]:
        """Load logging configuration from file or use defaults."""
        default_config = {
            "logging": {
                "level": "INFO",
                "file_path": "data/logs/policypal.log",
                "rotation": "10 MB",
                "retention": "30 days",
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                return default_config
        
        return default_config
    
    def _setup_logger(self):
        """Configure loguru logger with custom settings."""
        # Remove default handler
        logger.remove()
        
        # Get config
        log_config = self.config.get("logging", {})
        level = log_config.get("level", "INFO")
        file_path = log_config.get("file_path", "data/logs/policypal.log")
        rotation = log_config.get("rotation", "10 MB")
        retention = log_config.get("retention", "30 days")
        format_str = log_config.get("format", 
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Ensure log directory exists
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add console handler
        logger.add(
            sys.stdout,
            format=format_str,
            level=level,
            colorize=True
        )
        
        # Add file handler
        logger.add(
            file_path,
            format=format_str,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
        
        logger.info("PolicyPal logger initialized successfully")
    
    def get_logger(self, name: str = "PolicyPal"):
        """
        Get a logger instance with the specified name.
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        return logger.bind(name=name)


# Global logger instance
_policy_pal_logger: Optional[PolicyPalLogger] = None


def get_logger(name: str = "PolicyPal"):
    """
    Get the global PolicyPal logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    global _policy_pal_logger
    
    if _policy_pal_logger is None:
        _policy_pal_logger = PolicyPalLogger()
    
    return _policy_pal_logger.get_logger(name)


def setup_logging(config_path: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        config_path: Path to logging configuration file
    """
    global _policy_pal_logger
    _policy_pal_logger = PolicyPalLogger(config_path)


# Convenience functions for common logging patterns
def log_function_call(func_name: str, args: Optional[list[Any]] = None, kwargs: Optional[dict[str, Any]] = None):
    """Log function call with parameters."""
    logger_instance = get_logger()
    params = []
    if args:
        params.extend([str(arg) for arg in args])
    if kwargs:
        params.extend([f"{k}={v}" for k, v in kwargs.items()])
    
    logger_instance.debug(f"Calling {func_name}({', '.join(params)})")


def log_function_result(func_name: str, result: Optional[Any] = None, error: Optional[str] = None):
    """Log function result or error."""
    logger_instance = get_logger()
    if error:
        logger_instance.error(f"Function {func_name} failed: {error}")
    else:
        logger_instance.debug(f"Function {func_name} completed successfully")


def log_performance(operation: str, duration: float, details: str = ""):
    """Log performance metrics."""
    logger_instance = get_logger()
    logger_instance.info(f"Performance: {operation} took {duration:.2f}s {details}")


def log_data_processing(operation: str, input_size: int, output_size: Optional[int] = None):
    """Log data processing operations."""
    logger_instance = get_logger()
    if output_size:
        logger_instance.info(f"Data processing: {operation} - Input: {input_size}, Output: {output_size}")
    else:
        logger_instance.info(f"Data processing: {operation} - Input: {input_size}")


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Get logger
    logger_instance = get_logger("test")
    
    # Test different log levels
    logger_instance.debug("This is a debug message")
    logger_instance.info("This is an info message")
    logger_instance.warning("This is a warning message")
    logger_instance.error("This is an error message")
    
    # Test performance logging
    log_performance("test_operation", 1.23, "with 100 documents")
    
    # Test data processing logging
    log_data_processing("document_parsing", 10, 8) 