"""Type aliases used throughout the package."""
from typing import Union
import numpy as np

FloatArray = np.ndarray  # dtype float64, any shape
Float1D = np.ndarray     # shape (N,)
Float2D = np.ndarray     # shape (M, N)
IntOrNone = Union[int, None]
SeedType = Union[int, np.random.Generator, None]
