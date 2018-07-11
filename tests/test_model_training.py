from random import random
from unittest import TestCase

from numpy import median

from examples.trivial_model import TrivialModel


class TestModelTraining(TestCase):
    @staticmethod
    def _train_test_split(func, **kwargs):
        model, score = TrivialModel.train(
            TrivialModel.features_from_list((n, func(n)) for n in range(20)),
            **kwargs
        )

        return score

    def test_train_test_split_perfect(self):
        self.assertEqual(
            self._train_test_split(lambda n: 10 * n, train_test_split_ratio=0.2),
            1
        )

    def test_train_test_split_noisy(self):
        for n in range(1, 10):
            with self.subTest(train_test_split_ratio=n / 10):
                score = median([
                    self._train_test_split(lambda m: m + 5 * random(), train_test_split_ratio=n / 10)
                    for _ in range(5)
                ])

                self.assertGreater(score, 0)
                self.assertLess(score, 1)

    def test_train_test_split_pure_noise(self):
        score = median([
            self._train_test_split(lambda n: random(), train_test_split_ratio=0.2)
            for _ in range(5)
        ])
        self.assertLess(score, 0)

    def test_train_test_split_errors(self):
        for train_test_split_ratio in (-1, -0.5, 0, 1, 1.5, 2):
            with self.subTest(train_test_split_ratio=train_test_split_ratio), \
                 self.assertRaises(ValueError):
                self._train_test_split(lambda n: 10 * n, train_test_split_ratio=train_test_split_ratio)

        for test_sample_count in (-1, 0):
            with self.subTest(test_sample_count=test_sample_count), \
                 self.assertRaises(ValueError):
                self._train_test_split(lambda n: 10 * n, test_sample_count=test_sample_count)
