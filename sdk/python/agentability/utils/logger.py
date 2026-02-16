"""Structured logging utility for Agentability.

Follows Google Cloud Logging standards with structured JSON output.
"""

import logging
import sys
from typing import Any, Dict, Optional


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes."""
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


class AgentabilityFormatter(logging.Formatter):
    """Custom formatter with colors and structured output."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.GRAY,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.MAGENTA,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        
        # Format: [LEVEL] module: message
        formatted = (
            f"{level_color}[{record.levelname}]{Colors.RESET} "
            f"{Colors.CYAN}{record.name}{Colors.RESET}: "
            f"{record.getMessage()}"
        )
        
        # Add exception info if present
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        
        return formatted


# Global logger configuration
_loggers: Dict[str, logging.Logger] = {}
_default_level = logging.INFO


def configure_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    use_colors: bool = True,
) -> None:
    """Configure global logging settings.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (if not using default)
        use_colors: Whether to use colored output
    """
    global _default_level
    _default_level = level
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Set formatter
    if use_colors:
        handler.setFormatter(AgentabilityFormatter())
    else:
        formatter = logging.Formatter(
            format_string or "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
    
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger for a module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    if name not in _loggers:
        logger = logging.getLogger(name)
        logger.setLevel(_default_level)
        _loggers[name] = logger
    
    return _loggers[name]


# Initialize default logging on import
configure_logging()
