from sweetPeaBush.utils import get_lag
import itertools
import math
from sweetPeaBush.utils import get_lag, ValidationErrorCollection, ValidationError


def validation_errors_crossings(block, df, kind="regular_only"):
    validation_errors = ValidationErrorCollection()
    if hasattr(block, "crossing"):
        crossings = [block.crossing]
    else:
        crossings = block.crossings

    for crossing in crossings:
        column_names = [f.name for f in crossing]
        lag = max([get_lag(f) for f in crossing]) - 1
        _df = df.copy()[lag:]
        # get each combination of each level in each factor
        level_name_lists = [[l.name for l in f.levels] for f in crossing]
        level_weight_list = [[l.weight for l in f.levels] for f in crossing]
        # get each combination
        level_combinations = list(itertools.product(*level_name_lists))
        level_weight_combinations = list(itertools.product(*level_weight_list))

        expected_counts = [math.prod(comb) for comb in level_weight_combinations]
        counts = [(df[column_names].eq(combo).all(axis=1)).sum() for combo in level_combinations]

        if max(expected_counts) <= 0 or max(counts) <= 0:
            raise ValueError('Max counts in crossing is 0')
        expected_counts_normalized = [e_c / max(expected_counts) for e_c in expected_counts]
        counts_normalized = [c / max(counts) for c in counts]
        for i, (e_c, c) in enumerate(zip(expected_counts_normalized, counts_normalized)):
            if not (math.isclose(e_c, c)):
                validation_errors.add_error(
                    ValidationError(
                        'crossing', str(column_names), level_combinations[i]
                    )
                )
        return validation_errors
