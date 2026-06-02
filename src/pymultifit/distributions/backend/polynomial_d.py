"""Created on Jul 12 05:01:19 2025"""

from __future__ import annotations

from . import BaseDistribution
from ..utilities_d import cubic, line, quadratic
from ...typing import ArrayLike, NDArray


class LineFunction(BaseDistribution):
    """Wrapper class for a linear function :math:`y = mx + b`.

    Parameters
    ----------
    slope :
        The slope parameter :math:`m`. Defaults to 1.0.
    intercept :
        The intercept parameter :math:`b`. Defaults to 1.0.
    normalize :
        Accepted for API consistency; has no effect. Defaults to ``False``.
    """

    def __init__(self, slope: float = 1.0, intercept: float = 1.0, normalize: bool = False):
        """
        Initialize LineFunction.

        Parameters
        ----------
        slope :
            The slope parameter :math:`m`. Defaults to 1.0.
        intercept :
            The intercept parameter :math:`b`. Defaults to 1.0.
        normalize :
            Accepted for API consistency; has no effect. Defaults to ``False``.
        """
        self.slope = slope
        self.intercept = intercept

        self.norm = normalize

    def pdf(self, x: ArrayLike) -> NDArray:
        """Calculates the line function.

        Parameters
        ----------
        x :
            Input array of values.

        Returns
        -------
        NDArray
            Array of the same shape as :math:`x`, containing the evaluated values.
        """
        return line(x, slope=self.slope, intercept=self.intercept)


class QuadraticFunction(BaseDistribution):
    """Wrapper class for a quadratic function :math:`y = ax^2 + bx + c`.

    Parameters
    ----------
    a :
        The quadratic coefficient :math:`a`. Defaults to 1.0.
    b :
        The linear coefficient :math:`b`. Defaults to 1.0.
    c :
        The constant coefficient :math:`c`. Defaults to 1.0.
    normalize :
        Accepted for API consistency; has no effect. Defaults to ``False``.
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0, normalize: bool = False):
        """
        Initialize QuadraticFunction.

        Parameters
        ----------
        a :
            The quadratic coefficient :math:`a`. Defaults to 1.0.
        b :
            The linear coefficient :math:`b`. Defaults to 1.0.
        c :
            The constant coefficient :math:`c`. Defaults to 1.0.
        normalize :
            Accepted for API consistency; has no effect. Defaults to ``False``.
        """
        self.a = a
        self.b = b
        self.c = c

        self.norm = normalize

    def pdf(self, x: ArrayLike) -> NDArray:
        """Calculates the quadratic function.

        Parameters
        ----------
        x :
            Input array of values.

        Returns
        -------
        NDArray
            Array of the same shape as :math:`x`, containing the evaluated values.
        """
        return quadratic(x, a=self.a, b=self.b, c=self.c)


class CubicFunction(BaseDistribution):
    """Wrapper class for a cubic function :math:`y = ax^3 + bx^2 + cx + d`.

    Parameters
    ----------
    a :
        The cubic coefficient :math:`a`. Defaults to 1.0.
    b :
        The quadratic coefficient :math:`b`. Defaults to 1.0.
    c :
        The linear coefficient :math:`c`. Defaults to 1.0.
    d :
        The constant coefficient :math:`d`. Defaults to 1.0.
    normalize :
        Accepted for API consistency; has no effect. Defaults to ``False``.
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0, d: float = 1.0, normalize: bool = False):
        """
        Initialize CubicFunction.

        Parameters
        ----------
        a :
            The cubic coefficient :math:`a`. Defaults to 1.0.
        b :
            The quadratic coefficient :math:`b`. Defaults to 1.0.
        c :
            The linear coefficient :math:`c`. Defaults to 1.0.
        d :
            The constant coefficient :math:`d`. Defaults to 1.0.
        normalize :
            Accepted for API consistency; has no effect. Defaults to ``False``.
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.norm = normalize

    def pdf(self, x: ArrayLike) -> NDArray:
        """Calculates the cubic function.

        Parameters
        ----------
        x :
            Input array of values.

        Returns
        -------
        NDArray
            Array of the same shape as :math:`x`, containing the evaluated values.
        """
        return cubic(x, a=self.a, b=self.b, c=self.c, d=self.d)
