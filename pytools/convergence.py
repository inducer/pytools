from typing import List, Optional, Tuple

import numpy as np


# {{{ eoc estimation --------------------------------------------------------------

def estimate_order_of_convergence(abscissae, errors):
    """Assuming that abscissae and errors are connected by a law of the form

    error = constant * abscissa ^ (order),

    this function finds, in a least-squares sense, the best approximation of
    constant and order for the given data set. It returns a tuple (constant, order).
    """
    assert len(abscissae) == len(errors)
    if len(abscissae) <= 1:
        raise RuntimeError("Need more than one value to guess order of convergence.")

    coefficients = np.polyfit(np.log10(abscissae), np.log10(errors), 1)
    return 10**coefficients[-1], coefficients[-2]


class EOCRecorder:
    """
    .. automethod:: add_data_point

    .. automethod:: estimate_order_of_convergence
    .. automethod:: order_estimate
    .. automethod:: max_error

    .. automethod:: pretty_print
    .. automethod:: write_gnuplot_file
    """

    def __init__(self):
        self.history: List[Tuple[float, float]] = []

    def add_data_point(self, abscissa: float, error: float) -> None:
        if not (np.isscalar(abscissa)
                or (isinstance(abscissa, np.ndarray) and abscissa.shape == ())):
            raise TypeError(
                    f"'abscissa' is not a scalar: '{type(abscissa).__name__}'")

        if not (np.isscalar(error)
                or (isinstance(error, np.ndarray) and error.shape == ())):
            raise TypeError(f"'error' is not a scalar: '{type(error).__name__}'")

        self.history.append((abscissa, error))

    def estimate_order_of_convergence(self,
            gliding_mean: Optional[int] = None,
            ) -> np.ndarray:
        abscissae = np.array([a for a, e in self.history])
        errors = np.array([e for a, e in self.history])

        # NOTE: in case any of the errors are exactly 0.0, which
        # can give NaNs in `estimate_order_of_convergence`
        emax = np.amax(errors)
        errors += (1 if emax == 0 else emax) * np.finfo(errors.dtype).eps

        size = len(abscissae)
        if gliding_mean is None:
            gliding_mean = size

        data_points = size - gliding_mean + 1
        result = np.zeros((data_points, 2), float)
        for i in range(data_points):
            result[i, 0], result[i, 1] = estimate_order_of_convergence(
                abscissae[i:i+gliding_mean], errors[i:i+gliding_mean])
        return result

    def order_estimate(self) -> float:
        return self.estimate_order_of_convergence()[0, 1]

    def max_error(self) -> float:
        return max(err for absc, err in self.history)

    def _to_table(self, *,
            abscissa_label="h",
            error_label="Error",
            gliding_mean=2,
            abscissa_format="%s",
            error_format="%s",
            eoc_format="%s"):
        from pytools import Table

        tbl = Table()
        tbl.add_row((abscissa_label, error_label, "Running EOC"))

        gm_eoc = self.estimate_order_of_convergence(gliding_mean)
        for i, (absc, err) in enumerate(self.history):
            absc_str = abscissa_format % absc
            err_str = error_format % err
            if i < gliding_mean-1:
                eoc_str = ""
            else:
                eoc_str = eoc_format % (gm_eoc[i - gliding_mean + 1, 1])

            tbl.add_row((absc_str, err_str, eoc_str))

        return tbl

    def pretty_print(self, *,
            abscissa_label: str = "h",
            error_label: str = "Error",
            gliding_mean: int = 2,
            abscissa_format: str = "%s",
            error_format: str = "%s",
            eoc_format: str = "%s",
            table_type: str = "markdown") -> str:
        tbl = self._to_table(
                abscissa_label=abscissa_label, error_label=error_label,
                abscissa_format=abscissa_format,
                error_format=error_format,
                eoc_format=eoc_format,
                gliding_mean=gliding_mean)

        if table_type == "markdown":
            tbl_str = tbl.github_markdown()
        elif table_type == "latex":
            tbl_str = tbl.latex()
        elif table_type == "ascii":
            tbl_str = str(tbl)
        elif table_type == "csv":
            tbl_str = tbl.csv()
        else:
            raise ValueError(f"unknown table type: {table_type}")

        if len(self.history) > 1:
            return "{}\n\nOverall EOC: {}".format(tbl_str,
                    self.estimate_order_of_convergence()[0, 1])
        else:
            return tbl_str

    def __str__(self):
        return self.pretty_print()

    def write_gnuplot_file(self, filename: str) -> None:
        outfile = open(filename, "w")
        for absc, err in self.history:
            outfile.write(f"{absc:f} {err:f}\n")
        result = self.estimate_order_of_convergence()
        const = result[0, 0]
        order = result[0, 1]
        outfile.write("\n")
        for absc, _err in self.history:
            outfile.write(f"{absc:f} {const * absc**(-order):f}\n")

# }}}


# {{{ p convergence verifier

class PConvergenceVerifier:
    def __init__(self):
        self.orders = []
        self.errors = []

    def add_data_point(self, order, error):
        self.orders.append(order)
        self.errors.append(error)

    def __str__(self):
        from pytools import Table
        tbl = Table()
        tbl.add_row(("p", "error"))

        for p, err in zip(self.orders, self.errors):
            tbl.add_row((str(p), str(err)))

        return str(tbl)

    def __call__(self):
        orders = np.array(self.orders, np.float64)
        errors = np.abs(np.array(self.errors, np.float64))

        rel_change = np.diff(1e-20 + np.log10(errors)) / np.diff(orders)

        assert (rel_change < -0.2).all()

# }}}


# vim: foldmethod=marker
