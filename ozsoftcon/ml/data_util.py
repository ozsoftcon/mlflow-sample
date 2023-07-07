from typing import Tuple
from numpy import array, loadtxt
from numpy.random import seed, shuffle


def read_ml_data(
        filepath: str,
        test_fraction: float,
        validation_fraction: float,
        seed_value: int = 42
) -> Tuple[array, array, array]:

    assert (test_fraction + validation_fraction) < 0.5

    seed(seed_value)

    total_data = loadtxt(filepath, delimiter=",", skiprows=1)

    total_rows_read = total_data.shape[0]

    row_indices = [idx for idx in range(total_rows_read)]
    shuffle(row_indices)

    test_rows = int(total_rows_read * test_fraction)
    validation_rows = int(total_rows_read * validation_fraction)

    test_row_indices = row_indices[:test_rows]
    validation_row_indices = row_indices[test_rows:(test_rows+validation_rows)]
    training_rows_indices = row_indices[(test_rows+validation_rows):]

    training_data = total_data[array(training_rows_indices), :]
    validation_data = total_data[array(validation_row_indices), :]
    test_data = total_data[array(test_row_indices), :]

    return (
        training_data,
        validation_data,
        test_data
    )