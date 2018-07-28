from unittest import TestCase

from smart_fruit import Model
from smart_fruit.feature_types import Number, Label, Vector


class TestCompoundTypes(TestCase):
    def test_vector(self):
        class ExampleModel(Model):
            class Input:
                a = Number()

            class Output:
                b = Vector([
                    Number(),
                    Label(['a', 'b']),
                    Number()
                ])

        samples = [
            (0, (1, 'a', 2)),
            (1, (3, 'b', 4))
        ]

        model = ExampleModel.train(ExampleModel.features_from_list(samples))

        predictions = model.predict(ExampleModel.input_features_from_list([[0], [1]]))

        for (sample_input, sample_output), prediction in zip(samples, predictions):
            with self.subTest(sample=sample_input):
                self.assertAlmostEqual(sample_output[0], prediction.b[0])
                self.assertEqual(sample_output[1], prediction.b[1])
                self.assertAlmostEqual(sample_output[2], prediction.b[2])

    def test_nested_vectors(self):
        class ExampleModel(Model):
            class Input:
                a = Number()

            class Output:
                b = Vector([
                    Number(),
                    Vector([
                        Number(),
                        Label(['a', 'b'])
                    ]),
                    Number()
                ])

        samples = [
            (0, (1, (2, 'a'), 3)),
            (1, (4, (5, 'b'), 6))
        ]

        model = ExampleModel.train(ExampleModel.features_from_list(samples))

        predictions = model.predict(ExampleModel.input_features_from_list([[0], [1]]))

        for (sample_input, sample_output), prediction in zip(samples, predictions):
            with self.subTest(sample=sample_input):
                self.assertAlmostEqual(sample_output[0], prediction.b[0])
                self.assertAlmostEqual(sample_output[1][0], prediction.b[1][0])
                self.assertEqual(sample_output[1][1], prediction.b[1][1])
                self.assertAlmostEqual(sample_output[2], prediction.b[2])

    def test_vector_validation(self):
        feature_type = Vector([
            Number(),
            Label(['a', 'b'])
        ])

        for a in ((1, 'a'), (3, 'b'), (17, 'a')):
            with self.subTest(a=a):
                self.assertEqual(feature_type.validate(a), a)

        for a in ((1,), (1, 'a', 2), ('a', 1), (1, 1)):
            with self.subTest(a=a), \
                 self.assertRaises((TypeError, ValueError)):
                feature_type.validate(a)
