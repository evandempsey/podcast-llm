"""
Utilities for checkpointing and resuming long-running processes.

This module provides functionality for saving and loading intermediate computation
results to disk, enabling efficient resumption of processing from the last successful
checkpoint. This is particularly useful for long-running podcast generation tasks
that may need to be interrupted and resumed.

Key components:
- Checkpointer: A class that manages saving/loading of checkpoint data with configurable
  paths and serialization
- to_snake_case: Helper function for converting checkpoint names to valid filenames

The checkpointing system helps with:
- Saving intermediate results during multi-step processing
- Resuming interrupted processes without recomputing completed steps  
- Debugging by examining saved checkpoint states
- Reducing wasted computation on process restarts

The module uses pickle for serialization by default but is designed to be extensible
to other serialization formats as needed.
"""


import logging
from typing import Any, Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
import pickle


logger = logging.getLogger(__name__)


def to_snake_case(text: str) -> str:
    """
    Convert a string to snake_case format.
    
    Takes any string input and converts it to snake_case by:
    1. Replacing spaces and hyphens with underscores
    2. Converting to lowercase
    3. Removing any non-alphanumeric characters except underscores
    
    Args:
        text (str): Input string to convert
        
    Returns:
        str: Snake case formatted string
    """
    # Replace spaces and hyphens with underscores
    text = text.replace(' ', '_').replace('-', '_')
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove any characters that aren't alphanumeric or underscore
    text = ''.join(c for c in text if c.isalnum() or c == '_')
    
    # Replace multiple consecutive underscores with single underscore
    while '__' in text:
        text = text.replace('__', '_')
        
    # Remove leading/trailing underscores
    return text.strip('_')



class Checkpointer:
    """
    A class for managing checkpointing of intermediate results during processing.

    The Checkpointer allows saving and loading of intermediate computation results to disk,
    enabling resumption of long-running processes from the last successful checkpoint.
    
    Key features:
    - Configurable checkpoint directory and key prefix for files
    - Can be enabled/disabled via constructor
    - Automatically creates checkpoint directory if needed
    - Saves results as pickle files with stage-specific names
    - Loads from existing checkpoints when available
    
    Example usage:
        checkpointer = Checkpointer(
            checkpoint_key='my_process_',
            enabled=True
        )
        
        # Will save result to disk and return it
        result = checkpointer.checkpoint(
            expensive_computation(), 
            stage_name='stage1'
        )
        
        # On subsequent runs, will load from disk instead of recomputing
        result = checkpointer.checkpoint(
            expensive_computation(),
            stage_name='stage1'
        )
    """
    def __init__(self, checkpoint_key: str, checkpoint_dir: str = '.checkpoints', enabled: bool = True):
        """
        Initialize the Checkpointer.

        Args:
            checkpoint_key (str): Base key to use for checkpoint filenames
            checkpoint_dir (str): Directory path for storing checkpoints
            enabled (bool): Whether to enable checkpointing functionality
        """
        logger.info(f"Initializing checkpointer with key: {checkpoint_key}")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.enabled = enabled
        self.checkpoint_key = checkpoint_key
        if enabled:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint(self, fn: Callable, args: list, stage_name: str = 'result') -> Any:
        if not self.enabled:
            return fn(*args)

        # Generate checkpoint filename using base key
        checkpoint_file = self.checkpoint_dir / f'{self.checkpoint_key}_{stage_name}.pkl'

        # Try to load from checkpoint
        if checkpoint_file.exists():
            logger.info(f'Loading checkpoint from {checkpoint_file}')
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        
        # If it doesn't exist, call the function
        result = fn(*args)

        # Save checkpoint
        logger.info(f'Saving checkpoint to {checkpoint_file}')
        with NamedTemporaryFile(dir=self.checkpoint_dir, delete=False) as f:
            temporary_file = Path(f.name)
            try:
                pickle.dump(result, f)
                f.close()
                temporary_file.replace(checkpoint_file)
            finally:
                temporary_file.unlink(missing_ok=True)

        return result
