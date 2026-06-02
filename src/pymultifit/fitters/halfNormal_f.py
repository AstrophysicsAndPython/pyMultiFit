"""Created on Dec 04 23:29:32 2024"""

import numpy as np

from ..distributions.utilities_d import half_normal_pdf_
from ..typing import ArrayLike
from .backend import BaseFitter
from .utilities_f import sanity_check


class HalfNormalFitter(BaseFitter):
    """A class for fitting multiple HalfNormal distributions to the given data."""

    def __init__(self, x_values: ArrayLike, y_values: ArrayLike, max_iterations: int = 1000):
        """
        Initialize HalfNormalFitter.

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
        """Return the parameter boundaries for the HalfNormal fitter.

        Returns
        -------
        tuple[tuple, tuple]
            Lower and upper bounds for ``(amplitude, std, loc)``.
        """
        lb = (0, 0, -np.inf)
        ub = (np.inf, np.inf, np.inf)
        return lb, ub

    @staticmethod
    def fitter(x, params: ArrayLike):
        """Evaluate the HalfNormal PDF at *x* given *params*.

        Parameters
        ----------
        x :
            The x-array at which the model is evaluated.
        params :
            Parameter array ``[amplitude, std, loc]``.

        Returns
        -------
        NDArray
            Evaluated y-values.
        """
        return half_normal_pdf_(x, *params)
