from unittest import TestCase

from smart_fruit import Model
from smart_fruit.feature_types import Number, Integer, Complex, Label


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

    def test_complex(self):
        class ExampleModel(Model):
            class Input:
                a = Number()

            class Output:
                b = Complex()

        model = ExampleModel.train(ExampleModel.features_from_list([
            (0, 1),
            (1, 1j)
        ]))

        predictions = model.predict(ExampleModel.input_features_from_list([[0], [0.5], [1]]))

        for a, b, output in zip((0, 0.5, 1), (1, 0.5 + 0.5j, 1j), predictions):
            with self.subTest(a=a):
                self.assertAlmostEqual(output.b, b)

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

    def test_multiple_types_in_single_model(self):
        class ExampleModel(Model):
            class Input:
                a = Number()
                b = Number()

            class Output:
                c = Number()
                d = Number()
                e = Label(['a', 'b'])
                f = Complex()
                g = Number()

        samples = [
            (0, 0, 0, 0, 'a', 0, 0),
            (0, 1, 0, 3, 'b', 1j, 0),
            (1, 0, 2, 0, 'a', 1, 0),
            (1, 1, 2, 3, 'b', 1 + 1j, 0)
        ]

        model = ExampleModel.train(ExampleModel.features_from_list(samples))

        predictions = model.predict(ExampleModel.input_features_from_list([
            sample[:2] for sample in samples
        ]))

        for sample, prediction in zip(samples, predictions):
            with self.subTest(sample=sample):
                self.assertAlmostEqual(sample[2], prediction.c)
                self.assertAlmostEqual(sample[3], prediction.d)
                self.assertEqual(sample[4], prediction.e)
                self.assertAlmostEqual(sample[5], prediction.f)
                self.assertAlmostEqual(sample[6], prediction.g)
