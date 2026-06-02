"""Created on Nov 30 11:30:45 2024"""

import numpy as np

from ..distributions.utilities_d import exponential_pdf_
from ..typing import ArrayLike
from .backend import BaseFitter
from .utilities_f import sanity_check


class ExponentialFitter(BaseFitter):
    """A class for fitting multiple Exponential distributions to the given data."""

    def __init__(self, x_values: ArrayLike, y_values: ArrayLike, max_iterations: int = 1000):
        """
        Initialize ExponentialFitter.

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
        self.pn_par = 2
        self.sn_par = {"loc": 0.0}

    def fit_boundaries(self):
        """Return the parameter boundaries for the Exponential fitter.

        Returns
        -------
        tuple[tuple, tuple]
            Lower and upper bounds for ``(amplitude, scale, loc)``.
        """
        lb = (0, 0, -np.inf)
        ub = (np.inf, np.inf, np.inf)
        return lb, ub

    @staticmethod
    def fitter(x, params: ArrayLike):
        """Evaluate the Exponential PDF at *x* given *params*.

        Parameters
        ----------
        x :
            The x-array at which the model is evaluated.
        params :
            Parameter array ``[amplitude, scale, loc]``.

        Returns
        -------
        NDArray
            Evaluated y-values.
        """
        return exponential_pdf_(x, *params)
