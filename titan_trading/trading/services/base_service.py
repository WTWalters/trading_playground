"""
Base service class for TITAN Trading System.

Provides common functionality for all services including
async/sync bridging, error handling, and logging.
"""
import asyncio
import logging
import functools
import traceback
from typing import Any, Callable, TypeVar, cast

# Type variables for better type hints
T = TypeVar('T')
AsyncFunc = TypeVar('AsyncFunc', bound=Callable[..., Any])


class BaseService:
    """
    Base class for service layer components.
    
    This class provides common functionality for all services:
    - Asynchronous function execution in a synchronous context
    - Error handling and logging
    - Resource cleanup
    
    All service classes should inherit from this base class.
    """
    
    def __init__(self):
        """Initialize the service with a class-specific logger."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initializing {self.__class__.__name__}")
        
    def run_async(self, async_func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Run an asynchronous function in a synchronous context.
        
        This method creates a new event loop, runs the provided async function
        to completion, and then closes the loop. It's used to bridge between
        Django's synchronous views and the async trading components.
        
        Args:
            async_func: The asynchronous function to run
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the asynchronous function
            
        Raises:
            Exception: Any exception raised by the async function
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            self.logger.debug(f"Running async function {async_func.__name__}")
            return loop.run_until_complete(async_func(*args, **kwargs))
        except Exception as e:
            self.logger.error(f"Error in async function {async_func.__name__}: {e}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            loop.close()
            
    @staticmethod
    def sync_wrap(async_func: AsyncFunc) -> AsyncFunc:
        """
        Decorator to create a synchronous wrapper for an async function.
        
        This decorator wraps an async method to make it callable from
        synchronous code. It maintains the original function signature
        and docstring.
        
        Args:
            async_func: The asynchronous method to wrap
            
        Returns:
            A synchronous wrapper function with the same signature
            
        Example:
            @BaseService.sync_wrap
            async def get_data(self, symbol):
                # Async implementation
                pass
        """
        @functools.wraps(async_func)
        def wrapper(self: BaseService, *args: Any, **kwargs: Any) -> Any:
            return self.run_async(async_func, self, *args, **kwargs)
        return cast(AsyncFunc, wrapper)
    
    async def _initialize_resources(self) -> None:
        """
        Initialize resources needed by the service.
        
        This is an abstract method that should be implemented by
        subclasses to initialize any required resources like database
        connections or API clients.
        """
        pass
    
    async def _cleanup_resources(self) -> None:
        """
        Clean up resources used by the service.
        
        This is an abstract method that should be implemented by
        subclasses to clean up any resources that need proper closure,
        like database connections or open files.
        """
        pass
    
    def cleanup(self) -> None:
        """
        Synchronous method to clean up resources.
        
        This method should be called when the service is no longer needed
        to ensure all resources are properly cleaned up.
        """
        self.run_async(self._cleanup_resources)
        self.logger.info(f"Cleaned up resources for {self.__class__.__name__}")
