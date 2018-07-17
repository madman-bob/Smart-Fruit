from io import StringIO
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

    invalid_iterable_inputs = (
        ('a', 'a'),
        (float('nan'), 'a'),
        (float('inf'), 'a'),
        (- float('inf'), 'a'),

        (1, 1),
        (1, 'd')
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

        with self.subTest(func=self.ExampleModel.input_features_from_list):
            features = list(self.ExampleModel.input_features_from_list([self.example_input]))

            self.assertEqual(features[0].number, 1)
            self.assertEqual(features[0].label, 'a')

        with self.subTest(func=self.ExampleModel.features_from_list):
            features = list(self.ExampleModel.features_from_list([self.example_input + self.example_output]))

            self.assertEqual(features[0][0].number, 1)
            self.assertEqual(features[0][0].label, 'a')

            self.assertEqual(features[0][1].number_a, 1)
            self.assertEqual(features[0][1].number_b, 2)

    def test_json_deserialization(self):
        with self.subTest(feature_type=self.ExampleModel.Input):
            feature = self.ExampleModel.Input.from_json(self.example_json_input)

            self.assertEqual(feature.number, 1)
            self.assertEqual(feature.label, 'a')

        with self.subTest(feature_type=self.ExampleModel.Output):
            feature = self.ExampleModel.Output.from_json(self.example_json_output)

            self.assertEqual(feature.number_a, 1)
            self.assertEqual(feature.number_b, 2)

        with self.subTest(func=self.ExampleModel.input_features_from_json):
            features = list(self.ExampleModel.input_features_from_json([self.example_json_input]))

            self.assertEqual(features[0].number, 1)
            self.assertEqual(features[0].label, 'a')

        with self.subTest(func=self.ExampleModel.features_from_json):
            features = list(self.ExampleModel.features_from_json([
                {**self.example_json_input, **self.example_json_output}
            ]))

            self.assertEqual(features[0][0].number, 1)
            self.assertEqual(features[0][0].label, 'a')

            self.assertEqual(features[0][1].number_a, 1)
            self.assertEqual(features[0][1].number_b, 2)

    def test_csv_deserialization(self):
        with self.subTest(func=self.ExampleModel.input_features_from_csv):
            features = list(self.ExampleModel.input_features_from_csv(
                StringIO(",".join(map(str, self.example_input)))
            ))

            self.assertEqual(features[0].number, 1)
            self.assertEqual(features[0].label, 'a')

        with self.subTest(func=self.ExampleModel.features_from_csv):
            features = list(self.ExampleModel.features_from_csv(
                StringIO(",".join(map(str, self.example_input + self.example_output)))
            ))

            self.assertEqual(features[0][0].number, 1)
            self.assertEqual(features[0][0].label, 'a')

            self.assertEqual(features[0][1].number_a, 1)
            self.assertEqual(features[0][1].number_b, 2)

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

    def test_value_coercion(self):
        self.assertEqual(
            self.ExampleModel.Input('1', 'a').validate(),
            self.ExampleModel.Input(1, 'a')
        )

    def test_invalid_deserialization(self):
        for invalid_input in self.invalid_iterable_inputs:
            with self.subTest(invalid_input=invalid_input), \
                 self.assertRaises((TypeError, ValueError)):
                self.ExampleModel.Input(*invalid_input).validate()
