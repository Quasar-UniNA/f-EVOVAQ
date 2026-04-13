import numpy as np
from typing import Callable, Optional, Sequence, Tuple, Union


ArrayLike = Union[np.ndarray, "cp.ndarray"]


class Problem:
    """
    Define a minimization problem for evolutionary optimization.

    Supports CPU (NumPy) and GPU (CuPy) backends transparently.

    Args:
        n_params: Number of real parameters to be optimized.
        obj_function: Objective function f(x, **args) -> float or vector.
                      Must accept x of shape (n_params,) or (pop_size, n_params).
        param_bounds: Tuple or list of (min, max) bounds.
        init_range: Initialization range (same format as param_bounds). If None, the range (-1, 1) is used.
        backend: "cpu" or "gpu".
        vectorized: If True, obj_function already supports batch input of shape (pop_size, n_params).
    """

    def __init__(
        self,
        n_params: int,
        obj_function: Callable[[ArrayLike], Union[float, ArrayLike]],
        param_bounds: Optional[Union[Tuple, Sequence[Tuple]]] = None,
        init_range: Optional[Union[Tuple, Sequence[Tuple]]] = None,
        backend: Optional[str] = "cpu",
        vectorized: bool = False,
    ):
        self.n_params = self._validate_n_params(n_params)
        self.obj_function = self._validate_callable(obj_function)
        self.vectorized = vectorized

        self.xp = self._get_backend(backend)

        self.param_bounds = self._normalize_ranges(param_bounds, self.n_params, default=(None, None))
        self.init_range = self._normalize_ranges(init_range, self.n_params, default=(-1, 1))

        self.bounds_min, self.bounds_max = self._ranges_to_arrays(
            self.param_bounds,
            fill_min=-self.xp.inf,
            fill_max=self.xp.inf
        )

        self.init_min, self.init_max = self._ranges_to_arrays(
            self.init_range,
            fill_min=-1.0,
            fill_max=1.0
        )

    @staticmethod
    def _get_backend(backend: Optional[str]):
        """
        Set the backend for the computation.
        """
        if backend and backend.lower() == "gpu":
            try:
                import cupy as cp
                return cp
            except ImportError as e:
                raise ImportError(
                    "[f-EVOVAQ] CuPy is required for GPU backend. "
                    "Install it following https://docs.cupy.dev/"
                ) from e
        return np
    
    @staticmethod
    def _validate_n_params(n: int):
        """
        Validate the number of parameters in the problem.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("`n_params` must be a positive integer.")
        return n

    @staticmethod
    def _validate_callable(fn: Callable):
        """
        Validate the objective function to be a callable.
        """
        if not callable(fn):
            raise TypeError("`obj_function` must be callable.")
        return fn

    @staticmethod
    def _normalize_ranges(ranges, n_params, default):
        """
        Create an array of tuples of (min, max).
        """
        if ranges is None:
            return [default] * n_params

        if isinstance(ranges, tuple):
            return [ranges] * n_params

        if isinstance(ranges, list):
            if len(ranges) != n_params:
                raise ValueError("Length mismatch")
            return ranges

        raise TypeError("Invalid format")


    def _ranges_to_arrays(self, ranges, fill_min, fill_max):
        """
        Convert tuples of ranges to arrays of min and max.
        """
        _mins, _maxs = zip(*ranges)

        mins = [fill_min if m is None else m for m in _mins]
        maxs = [fill_max if m is None else m for m in _maxs]

        return (
            self.xp.array(mins, dtype=self.xp.float32),
            self.xp.array(maxs, dtype=self.xp.float32),
        )

    def generate_individual(self):
        """Generate a single random individual."""
        return self.xp.random.uniform(
            low=self.init_min,
            high=self.init_max,
            size=(self.n_params,),
        )

    def generate_random_pop(self, pop_size: int):
        """Generate a random population of individuals."""
        if pop_size <= 1:
            raise ValueError("pop_size must be > 1")

        return self.xp.random.uniform(
            low=self.init_min,
            high=self.init_max,
            size=(pop_size, self.n_params),
        )

    def evaluate_fitness(self, params: ArrayLike):
        """
        Evaluate objective function.

        Supports:
        - individual vector: (n_params,)
        - population vector: (pop_size, n_params)
        """

        params = self.xp.asarray(params)

        if params.ndim == 1:
            return self.obj_function(params)

        if params.ndim == 2:
            if self.vectorized:
                return self.obj_function(params)
            else:
                return self.xp.array(
                    [self.obj_function(p) for p in params]
                )

        raise ValueError("Invalid parameter shape.")

    def check_bounds(self, params: ArrayLike):
        """
        Clip parameters to bounds.
        """
        return self.xp.clip(params, self.bounds_min, self.bounds_max, out=params)

    def to_cpu(self, array: ArrayLike):
        """
        Convert to NumPy array.
        """
        if self.xp.__name__ == "cupy":
            return array.get()
        return array