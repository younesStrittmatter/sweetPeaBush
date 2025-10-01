import sweetpea as sp
from sweetTea.utils import get_lag, ValidationErrorCollection, ValidationError


def validation_errors_design(block, df, kind="regular_only"):
    validation_errors = ValidationErrorCollection()
    design = block.design
    regular_factors = list(filter(_is_regular_factor, design))
    within_factors = list(filter(_is_within_factor, design))
    transition_factors = list(filter(_is_transition_factor, design))

    if kind == "regular_only":
        _all_factor_levels_in_design(regular_factors, df, validation_errors)
        return validation_errors
    _all_factor_levels_in_design(regular_factors, df, validation_errors)
    _all_factor_levels_in_design(within_factors, df, validation_errors)
    for f in transition_factors:
        lag = get_lag(f)
        _all_factor_levels_in_design([f], df, validation_errors, lag - 1)
        _transition_fct_correct([f], df, validation_errors, lag - 1)
    _within_fct_correct(within_factors, df, validation_errors)
    return validation_errors



def _all_factor_levels_in_design(factors, df, validation_errors, lag=0):
    _df = df.copy()
    if lag:
        _df = _df[lag:]

    for factor in factors:
        if not factor.name in _df.columns:
            validation_errors.add_error(ValidationError(
                "FACTOR_NOT_IN_DATA", factor.name, None)
            )
            continue
        column = _df[factor.name]
        allowed_levels = [l.name for l in factor.levels]
        unique_values = list(set(column.values))
        # test if every unique value is an allowed value
        for v in unique_values:
            if not v in allowed_levels:
                validation_errors.add_error(ValidationError(
                    "DATA_LEVEL_NOT_ALLOWED", factor.name, v
                ))


def _within_fct_correct(factors, df, validation_errors, lag=0):
    _df = df.copy()
    if lag:
        _df = _df[lag:]
    for factor in factors:
        for index, row in df.iterrows():
            data_derived_level_prediction = ''
            for l in factor.levels:
                window = l.window
                window_factors = [f.name for f in window.factors]
                data_levels = [row[fn] for fn in window_factors]
                if window.predicate(*data_levels):
                    data_derived_level_prediction = l.name
                    break
            if data_derived_level_prediction == row[factor.name]:
                continue
            else:
                validation_errors.add_error(
                    ValidationError('DERIVED LEVEL PREDICATE ERROR',
                                    factor.name,
                                    row[factor.name],
                                    index))


def _transition_fct_correct(factors, df, validation_errors, lag=0):
    _df = df.copy()
    for factor in factors:
        for index, row in df.iterrows():
            if index < lag:
                continue
            data_derived_level_prediction = ''
            for l in factor.levels:
                window = l.window
                window_factors = [f.name for f in window.factors]
                data_levels = []
                for w_factor in window_factors:
                    level_current = [row[w_factor]]
                    additional_levels = []
                    for i in range(lag):
                        # add the previous entries before the row
                        additional_levels = [_df[w_factor][index - (i + 1)]] + additional_levels
                    data_levels.append(level_current + additional_levels)
                if window.predicate(*data_levels):
                    data_derived_level_prediction = l.name
                    break
            if data_derived_level_prediction == row[factor.name]:
                continue
            else:
                validation_errors.add_error(
                    ValidationError('DERIVED LEVEL PREDICATE ERROR',
                                    factor.name,
                                    row[factor.name],
                                    index))


def _is_derived_factor(factor):
    return any(isinstance(l, sp.DerivedLevel) for l in factor.levels)


def _is_within_factor(factor):
    if _is_derived_factor(factor):
        return all(isinstance(l.window, sp.WithinTrial) for l in factor.levels)
    return False


def _is_transition_factor(factor):
    return factor.has_complex_window


def _is_regular_factor(factor):
    return not _is_derived_factor(factor)
