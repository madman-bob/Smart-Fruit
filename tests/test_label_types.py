from enum import Enum
from unittest import TestCase

from pandas import Series

from smart_fruit import Model
from smart_fruit.feature_types import Label


class TestLabelTypes(TestCase):
    def _test_labels(self, labels):
        class ExampleModel(Model):
            class Input:
                label = Label(labels)

        with self.subTest("Label to pandas Series"):
            for i, label in enumerate(labels):
                self.assertTrue(
                    ExampleModel.Input.label.to_series(label).equals(
                        self._basis_vector(len(labels), i)
                    )
                )

        with self.subTest("Pandas Series to label"):
            for i, label in enumerate(labels):
                self.assertEqual(
                    ExampleModel.Input.label.from_series(self._basis_vector(len(labels), i)),
                    label
                )

    @staticmethod
    def _basis_vector(length, index):
        vector = Series([0 for _ in range(length)])
        vector[index] = 1
        return vector

    def test_string_labels(self):
        self._test_labels(['a', 'b', 'c'])

    def test_number_labels(self):
        self._test_labels([1, 17, 3.1415])

    def test_enum_labels(self):
        class Colours(Enum):
            red = 0
            green = 1
            blue = 2

        self._test_labels(Colours)
