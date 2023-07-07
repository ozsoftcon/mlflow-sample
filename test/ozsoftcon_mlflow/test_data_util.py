from unittest import TestCase, mock
from numpy.random import rand
from pytest import raises
from ozsoftcon.ml import read_ml_data


class TestMLDataUtil(TestCase):

    @mock.patch('ozsoftcon.ml.data_util.loadtxt')
    def test_data_split(self, mocked_loaded_data):

        mocked_loaded_data.return_value = rand(
            10, 3
        )

        train, validation, test = read_ml_data(
            "/some_path", test_fraction=0.1, validation_fraction=0.2
        )

        self.assertTupleEqual(train.shape, (7, 3))
        self.assertTupleEqual(validation.shape, (2, 3))
        self.assertTupleEqual(test.shape, (1, 3))

        mocked_loaded_data.reset()
        mocked_loaded_data.return_value = rand(
            12, 3
        )

        train, validation, test = read_ml_data(
            "/some_path", test_fraction=0.3, validation_fraction=0.1
        )

        self.assertTupleEqual(train.shape, (8, 3))
        self.assertTupleEqual(validation.shape, (1, 3))
        self.assertTupleEqual(test.shape, (3, 3))


    @mock.patch('ozsoftcon.ml.data_util.loadtxt')
    def test_data_split_fraction_check_fail(self, mocked_loaded_data):
        mocked_loaded_data.return_value = rand(
            10, 3
        )

        ## normally it is expected that test and validation are
        ## less than half of the total data
        with raises(AssertionError):
            _ = read_ml_data(
                "/some_path", test_fraction=0.2, validation_fraction=0.3
            )

