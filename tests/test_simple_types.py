from unittest import TestCase

from smart_fruit import Model
from smart_fruit.feature_types import Number, Integer, Label


class TestSimpleTypes(TestCase):
    def test_number(self):
        class ExampleModel(Model):
            class Input:
                a = Number()

            class Output:
                b = Number()

        model = ExampleModel.train(ExampleModel.features_from_list([
            (0, 0),
            (1, 10)
        ]))

        predictions = model.predict(ExampleModel.input_features_from_list([[0], [1]]))

        for n, output in zip((0, 1), predictions):
            with self.subTest(n=n):
                self.assertAlmostEqual(output.b, 10 * n)

    def test_integer(self):
        class ExampleModel(Model):
            class Input:
                a = Number()

            class Output:
                b = Integer()

        model = ExampleModel.train(ExampleModel.features_from_list([
            (0, 0),
            (1, 10)
        ]))

        predictions = model.predict(ExampleModel.input_features_from_list([[0], [0.01], [0.99], [1]]))

        for a, n, output in zip((0, 0.01, 0.99, 1), (0, 0, 10, 10), predictions):
            with self.subTest(a=a):
                self.assertEqual(output.b, n)

    def test_label(self):
        class ExampleModel(Model):
            class Input:
                a = Number()

            class Output:
                b = Label(['a', 'b'])

        model = ExampleModel.train(ExampleModel.features_from_list([
            (0, 'a'),
            (1, 'b')
        ]))

        predictions = model.predict(ExampleModel.input_features_from_list([[0], [1]]))

        for n, label, output in zip((0, 1), ('a', 'b'), predictions):
            with self.subTest(n=n, label=label):
                self.assertEqual(output.b, label)
