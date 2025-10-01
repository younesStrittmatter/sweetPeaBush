import warnings

def get_lag(factor):
    """
    Get the lag of a windowed factor
    :param factor: the factor to get the lag of
    :return: the lag of the factor

    Examples:
        >>> import sweetpea as sp
        >>> from src.sweetPeaBush.utils import get_lag
        >>> color = sp.Factor("color", ["red", "blue"])
        >>> get_lag(color)
        1
        >>> def is_lagging(color):
        ...     return color[0] != color[-2]
        >>> def is_not_lagging(color):
        ...     return not is_lagging(color)
        >>> lag = sp.DerivedLevel('lag', sp.Window(is_lagging, [color], 3, 1))
        >>> non_lag = sp.DerivedLevel('nonlag', sp.Window(is_not_lagging, [color], 3, 1))
        >>> lagging = sp.Factor('lagging', [lag, non_lag])
        >>> get_lag(lagging)
        3
    """

    if any([not hasattr(l, "window") for l in factor.levels]):
        return 1
    return max([l.window.width for l in factor.levels])

class ValidationError:
    def __init__(self, kind, factor_name, level_name=None, row=None):
        self.kind = kind
        self.factor = factor_name
        self.level = level_name
        self.row = row

    def __repr__(self):
        _str = f"ValidationError({self.kind}"
        if self.factor is not None:
            _str += f", Factor: {self.factor}"
        if self.level is not None:
            _str += f", Level: {self.level}"
        if self.row is not None:
            _str += f", Row: {self.row}"
        _str += ")"
        return _str

    def __str__(self):
        return self.__repr__()


class ValidationErrorCollection:
    def __init__(self):
        self.errors = []

    def add_error(self, error: ValidationError):
        self.errors.append(error)

    def __repr__(self):
        return self.errors.__repr__()

    def __str__(self):
        return self.__repr__()
