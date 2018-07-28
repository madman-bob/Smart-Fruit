from unittest import TestCase

from smart_fruit import Model
from smart_fruit.feature_types import Number, Integer, Complex, Label, Tag


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

    def test_number_validation(self):
        feature_type = Number()

        for n in (0, 1, 3.141592, -17):
            with self.subTest(n=n):
                self.assertEqual(feature_type.validate(n), n)

        for a, b in (("1", 1),):
            with self.subTest(a=a):
                self.assertEqual(feature_type.validate(a), b)

        for n in (1j, float("nan"), float("inf"), -float("inf"), "a"):
            with self.subTest(n=n), \
                 self.assertRaises((TypeError, ValueError)):
                feature_type.validate(n)

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

    def test_integer_validation(self):
        feature_type = Integer()

        for a, b in ((0, 0), (1, 1), (3.141592, 3), (-17, -17)):
            with self.subTest(a=a):
                self.assertEqual(feature_type.validate(a), b)

        for a in (1j, float("nan"), float("inf"), -float("inf"), "a"):
            with self.subTest(a=a), \
                 self.assertRaises((TypeError, ValueError)):
                feature_type.validate(a)

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

    def test_complex_validation(self):
        feature_type = Complex()

        for a in (0, 1, 3 + 4j, -1 + 7j):
            with self.subTest(a=a):
                self.assertEqual(feature_type.validate(a), a)

        for a in (float("nan"), float("inf"), -float("inf"), "a"):
            with self.subTest(a=a), \
                 self.assertRaises((TypeError, ValueError)):
                feature_type.validate(a)

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

    def test_label_validation(self):
        ob = object()
        bad_ob = object()

        feature_type = Label(['a', 'b', 0, ob])

        for a in ('a', 'b', 0, ob):
            with self.subTest(a=a):
                self.assertEqual(feature_type.validate(a), a)

        for a in ('c', 1, bad_ob):
            with self.subTest(a=a), \
                 self.assertRaises(TypeError):
                feature_type.validate(a)

    def test_tag_input(self):
        class ExampleModel(Model):
            class Input:
                a = Number()
                b = Tag()

            class Output:
                c = Number()

        sample_inputs = [
            (0, 0),
            (1, 1),
            (0, 1),
            (1, 0),
            (0, "a"),
            (0, object())
        ]

        model = ExampleModel.train(ExampleModel.features_from_list([
            (0, 0, 0),
            (1, 1, 1)
        ]))

        predictions = model.predict(ExampleModel.input_features_from_list(sample_inputs))

        for sample_input, prediction in zip(sample_inputs, predictions):
            with self.subTest(sample=sample_input):
                self.assertAlmostEqual(prediction.c, sample_input[0])

    def test_tag_output(self):
        class ExampleModel(Model):
            class Input:
                a = Number()

            class Output:
                b = Tag()
                c = Number()

        model = ExampleModel.train(ExampleModel.features_from_list([
            (0, 0, 0),
            (1, 1, 1)
        ]))

        with self.assertRaisesRegex(TypeError, "May not predict a Tag"):
            next(model.predict([ExampleModel.Input(0)]))

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
