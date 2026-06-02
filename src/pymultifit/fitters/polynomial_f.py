"""Created on Aug 10 23:37:54 2024"""

import numpy as np

from ..distributions.utilities_d import cubic, line, quadratic
from ..typing import ArrayLike
from .backend import BaseFitter
from .utilities_f import sanity_check


class LineFitter(BaseFitter):
    """A class for fitting one or more linear functions to the given data.

    Parameters
    ----------
    x_values :
        Input x-values for fitting.
    y_values :
        Input y-values for fitting.
    max_iterations :
        Maximum number of iterations for the fitting algorithm. Defaults to 1000.
    """

    def __init__(self, x_values: ArrayLike, y_values: ArrayLike, max_iterations: int = 1000):
        """
        Initialize LineFitter.

        Parameters
        ----------
        x_values :
            Input x-values for fitting.
        y_values :
            Input y-values for fitting.
        max_iterations :
            Maximum number of iterations for the fitting algorithm. Defaults to 1000.
        """
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 2

    def fit_boundaries(self):
        """Return the parameter boundaries for the line fitter.

        Returns
        -------
        tuple[tuple, tuple]
            Lower and upper bounds for the slope and intercept parameters.
        """
        lb = (-np.inf, -np.inf)
        ub = (np.inf, np.inf)

        return lb, ub

    @staticmethod
    def fitter(x, params: ArrayLike):
        """Evaluate a linear function at *x* given *params*.

        Parameters
        ----------
        x :
            The x-array at which the model is evaluated.
        params :
            Parameter array ``[slope, intercept]``.

        Returns
        -------
        NDArray
            Evaluated y-values.
        """
        return line(x, *params)


class QuadraticFitter(BaseFitter):
    """A class for fitting one or more quadratic functions to the given data.

    Parameters
    ----------
    x_values :
        Input x-values for fitting.
    y_values :
        Input y-values for fitting.
    max_iterations :
        Maximum number of iterations for the fitting algorithm. Defaults to 1000.
    """

    def __init__(self, x_values: ArrayLike, y_values: ArrayLike, max_iterations: int = 1000):
        """
        Initialize QuadraticFitter.

        Parameters
        ----------
        x_values :
            Input x-values for fitting.
        y_values :
            Input y-values for fitting.
        max_iterations :
            Maximum number of iterations for the fitting algorithm. Defaults to 1000.
        """
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 3

    def fit_boundaries(self):
        """Return the parameter boundaries for the quadratic fitter.

        Returns
        -------
        tuple[tuple, tuple]
            Lower and upper bounds for the three quadratic coefficients.
        """
        lb = (-np.inf, -np.inf, -np.inf)
        ub = (np.inf, np.inf, np.inf)

        return lb, ub

    @staticmethod
    def fitter(x, params: ArrayLike):
        """Evaluate a quadratic function at *x* given *params*.

        Parameters
        ----------
        x :
            The x-array at which the model is evaluated.
        params :
            Parameter array ``[a, b, c]``.

        Returns
        -------
        NDArray
            Evaluated y-values.
        """
        return quadratic(x, *params)


class CubicFitter(BaseFitter):
    """A class for fitting one or more cubic functions to the given data.

    Parameters
    ----------
    x_values :
        Input x-values for fitting.
    y_values :
        Input y-values for fitting.
    max_iterations :
        Maximum number of iterations for the fitting algorithm. Defaults to 1000.
    """

    def __init__(self, x_values: ArrayLike, y_values: ArrayLike, max_iterations: int = 1000):
        """
        Initialize CubicFitter.

        Parameters
        ----------
        x_values :
            Input x-values for fitting.
        y_values :
            Input y-values for fitting.
        max_iterations :
            Maximum number of iterations for the fitting algorithm. Defaults to 1000.
        """
        x_values, y_values = sanity_check(x_values=x_values, y_values=y_values)
        super().__init__(x_values=x_values, y_values=y_values, max_iterations=max_iterations)
        self.n_par = 4

    def fit_boundaries(self):
        """Return the parameter boundaries for the cubic fitter.

        Returns
        -------
        tuple[tuple, tuple]
            Lower and upper bounds for the four cubic coefficients.
        """
        lb = (-np.inf, -np.inf, -np.inf, -np.inf)
        ub = (np.inf, np.inf, np.inf, np.inf)

        return lb, ub

    @staticmethod
    def fitter(x, params: ArrayLike):
        """Evaluate a cubic function at *x* given *params*.

        Parameters
        ----------
        x :
            The x-array at which the model is evaluated.
        params :
            Parameter array ``[a, b, c, d]``.

        Returns
        -------
        NDArray
            Evaluated y-values.
        """
        return cubic(x, *params)
