from unittest import TestCase

from examples.trivial_model import TrivialModel


class TestTrivialModel(TestCase):
    def _test_exact_relationship(self, func):
        model = TrivialModel.train(TrivialModel.features_from_list(
            (n, func(n))
            for n in range(20)
        ))

        test_values = (-5, 3, 17, 30, 3.1415)
        predictions = model.predict(TrivialModel.Output(n) for n in test_values)

        for n, prediction in zip(test_values, predictions):
            with self.subTest(n=n):
                self.assertIsInstance(prediction, TrivialModel.Output)
                self.assertAlmostEqual(prediction.output, func(n))

    def test_linear_relationship(self):
        self._test_exact_relationship(lambda n: 10 * n)

    def test_affine_relationship(self):
        self._test_exact_relationship(lambda n: 10 * n + 1)

    def test_quadratic_relationship(self):
        model = TrivialModel.train(TrivialModel.features_from_list(
            (n, n ** 2)
            for n in range(20)
        ))

        test_values = (0, 3, 10, 17, 20)
        predictions = model.predict(TrivialModel.Output(n) for n in test_values)

        for n, prediction in zip(test_values, predictions):
            with self.subTest(n=n):
                self.assertIsInstance(prediction, TrivialModel.Output)

                # Check that it's not exact, but still in the ballpark
                self.assertNotAlmostEqual(prediction.output, n ** 2, places=0)
                self.assertAlmostEqual(prediction.output, n ** 2, places=-3)
