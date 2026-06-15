import sys
import logging
from loguru import logger
import json
from src.core.config import settings


class JSONFormatter:
    """Formatter to export logs to structured JSON format."""
    def __call__(self, record):
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "module": record["module"],
            "function": record["name"] + ":" + record["function"],
            "line": record["line"],
            "process": record["process"].id,
        }
        if record["extra"]:
            log_entry["extra"] = {k: v for k, v in record["extra"].items() if k != "json_format"}
            
        # Store serialized JSON inside the extra dict under a safe namespace
        record["extra"]["json_format"] = json.dumps(log_entry)
        
        # Return a safe, compiled format string that Loguru handles natively
        return "{extra[json_format]}\n"


def setup_logging():
    """Configures Loguru to intercept standard Python logs and output structured JSON."""
    # Remove existing handlers
    logger.remove()
    
    # Configure Loguru handler with JSON formatting
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format=JSONFormatter(),
        backtrace=True,
        diagnose=True,
    )
    
    # Intercept standard library logging messages
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            frame = logging.currentframe()
            depth = 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    # Apply InterceptHandler to external libraries to centralize logs
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging_logger = logging.getLogger(name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False
        
    logger.info("Logging structured system successfully initialized", log_level=settings.LOG_LEVEL)
