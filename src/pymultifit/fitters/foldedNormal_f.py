"""Created on Dec 04 23:24:55 2024"""

import numpy as np

from ..distributions.utilities_d import folded_normal_pdf_
from ..typing import ArrayLike
from .backend import BaseFitter
from .utilities_f import sanity_check


class FoldedNormalFitter(BaseFitter):
    """A class for fitting multiple FoldedNormal distributions to the given data."""

    def __init__(self, x_values: ArrayLike, y_values: ArrayLike, max_iterations: int = 1000):
        """
        Initialize FoldedNormalFitter.

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
        """Return the parameter boundaries for the FoldedNormal fitter.

        Returns
        -------
        tuple[list, list]
            Lower and upper bounds for ``(amplitude, mu, std)``.
        """
        lb = [0, -np.inf, 0]
        ub = [np.inf, np.inf, np.inf]
        return lb, ub

    @staticmethod
    def fitter(x, params: ArrayLike):
        """Evaluate the FoldedNormal PDF at *x* given *params*.

        Parameters
        ----------
        x :
            The x-array at which the model is evaluated.
        params :
            Parameter array ``[amplitude, mu, std]``.

        Returns
        -------
        NDArray
            Evaluated y-values.
        """
        return folded_normal_pdf_(x, *params)
