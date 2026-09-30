"""Type aliases used throughout the package."""
import numpy as np

FloatArray = np.ndarray  # dtype float64, any shape
Float1D = np.ndarray     # shape (N,)
Float2D = np.ndarray     # shape (M, N)
IntOrNone = int | None
SeedType = int | np.random.Generator | None
