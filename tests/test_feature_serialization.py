from itertools import product
from unittest import TestCase

from smart_fruit import Model
from smart_fruit.feature_types import Number, Label


class TestFeatureSerialization(TestCase):
    class ExampleModel(Model):
        class Input:
            number = Number()
            label = Label(['a', 'b', 'c'])

        class Output:
            number_a = Number()
            number_b = Number()

    example_input = (1, 'a')
    example_output = (1, 2)

    example_json_input = {'number': 1, 'label': 'a'}
    example_json_output = {'number_a': 1, 'number_b': 2}

    valid_iterable_inputs = (
        (1, 'a'),
        (1, 'b'),
        (2, 'a')
    )

    def test_iterable_deserialization(self):
        with self.subTest(feature_type=self.ExampleModel.Input):
            feature = self.ExampleModel.Input(*self.example_input)

            self.assertEqual(feature.number, 1)
            self.assertEqual(feature.label, 'a')

        with self.subTest(feature_type=self.ExampleModel.Output):
            feature = self.ExampleModel.Output(*self.example_output)

            self.assertEqual(feature.number_a, 1)
            self.assertEqual(feature.number_b, 2)

    def test_json_deserialization(self):
        with self.subTest(feature_type=self.ExampleModel.Input):
            feature = self.ExampleModel.Input.from_json(self.example_json_input)

            self.assertEqual(feature.number, 1)
            self.assertEqual(feature.label, 'a')

        with self.subTest(feature_type=self.ExampleModel.Output):
            feature = self.ExampleModel.Output.from_json(self.example_json_output)

            self.assertEqual(feature.number_a, 1)
            self.assertEqual(feature.number_b, 2)

    def test_feature_equality(self):
        for a, b in product(self.valid_iterable_inputs, repeat=2):
            with self.subTest(a=a, b=b):
                self.assertEqual(
                    self.ExampleModel.Input(*a) == self.ExampleModel.Input(*b),
                    a == b
                )

    def test_iterable_serialization(self):
        with self.subTest(feature_type=self.ExampleModel.Input):
            feature = self.ExampleModel.Input(*self.example_input)
            self.assertEqual(tuple(feature), self.example_input)

        with self.subTest(feature_type=self.ExampleModel.Output):
            feature = self.ExampleModel.Output(*self.example_output)
            self.assertEqual(tuple(feature), self.example_output)

    def test_json_serialization(self):
        with self.subTest(feature_type=self.ExampleModel.Input):
            feature = self.ExampleModel.Input(*self.example_input)
            self.assertEqual(dict(feature.to_json()), self.example_json_input)

        with self.subTest(feature_type=self.ExampleModel.Output):
            feature = self.ExampleModel.Output(*self.example_output)
            self.assertEqual(dict(feature.to_json()), self.example_json_output)
