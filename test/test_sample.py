
from unittest import TestCase
from ozsoftcon.mlflow_wrap import Sample


class TestSample(TestCase):

    def test_sample(self):

        sample = Sample()
        assert sample.msg == "This is a sample file"