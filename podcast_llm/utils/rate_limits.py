import logging
import time
from functools import wraps


logger = logging.getLogger(__name__)


def rate_limit_per_minute(max_requests_per_minute: int):
    """
    Decorator that adds per-minute rate limiting to a function.
    
    Args:
        max_requests_per_minute (int): Maximum number of requests allowed per minute
        
    Returns:
        Callable: Decorated function with rate limiting
    """
    def decorator(func):
        last_request_time = 0
        min_interval = 60.0 / max_requests_per_minute  # Time between requests in seconds
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_request_time
            current_time = time.time()
            time_since_last = current_time - last_request_time
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                time.sleep(sleep_time)
                
            last_request_time = time.time()
            return func(*args, **kwargs)
            
        return wrapper
    return decorator


def retry_with_exponential_backoff(max_retries: int, base_delay: float = 1.0):
    """
    Decorator that retries a function with exponential backoff when exceptions occur.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        base_delay (float): Initial delay between retries in seconds. Will be exponentially increased.
        
    Returns:
        Callable: Decorated function with retry logic
        
    Example:
        @retry_with_exponential_backoff(max_retries=3, base_delay=1.0)
        def flaky_function():
            # Function that may fail intermittently
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        raise last_exception
                    
                    logger.warning(
                        f'Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}. '
                        f'Retrying in {delay:.1f}s...'
                    )
                    logger.warning(f"Caught exception: {str(e)}")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    
            return None  # Should never reach here
        return wrapper
    return decorator