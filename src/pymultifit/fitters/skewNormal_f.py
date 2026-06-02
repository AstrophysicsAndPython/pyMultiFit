"""Created on Jul 18 13:54:03 2024"""

import numpy as np

from ..distributions.utilities_d import skew_normal_pdf_
from ..typing import ArrayLike
from .backend import BaseFitter
from .utilities_f import sanity_check


class SkewNormalFitter(BaseFitter):
    """A class for fitting multiple SkewNormal distributions to the given data."""

    def __init__(self, x_values: ArrayLike, y_values: ArrayLike, max_iterations: int = 1000):
        """
        Initialize SkewNormalFitter.

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
        """Return the parameter boundaries for the SkewNormal fitter.

        Returns
        -------
        tuple[tuple, tuple]
            Lower and upper bounds for ``(amplitude, mu, std, alpha)``.
        """
        lb = (0, -np.inf, -np.inf, 0)
        ub = (np.inf, np.inf, np.inf, np.inf)
        return lb, ub

    @staticmethod
    def fitter(x, params: ArrayLike):
        """Evaluate the SkewNormal PDF at *x* given *params*.

        Parameters
        ----------
        x :
            The x-array at which the model is evaluated.
        params :
            Parameter array ``[amplitude, mu, std, alpha]``.

        Returns
        -------
        NDArray
            Evaluated y-values.
        """
        return skew_normal_pdf_(x, *params)
