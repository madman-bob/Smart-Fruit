from io import StringIO
from unittest import TestCase

from smart_fruit.utils import csv_open


class TestCSVOpen(TestCase):
    test_csv_path = "test_utils/example_csv.csv"
    test_csv_columns = ('a', 'b', 'c')
    test_csv_response = [
        {'a': '1', 'b': '2', 'c': '3'},
        {'a': '4', 'b': '5', 'c': '6'},
        {'a': 'α', 'b': 'β', 'c': 'γ'}
    ]

    def test_opens_csv_paths(self):
        self.assertEqual(
            list(csv_open(self.test_csv_path, self.test_csv_columns)),
            self.test_csv_response
        )

    def test_opens_csv_file_handles(self):
        with open(self.test_csv_path, encoding='utf-8') as csv_file:
            self.assertEqual(
                list(csv_open(csv_file, self.test_csv_columns)),
                self.test_csv_response
            )

    def test_no_given_columns(self):
        self.assertEqual(
            list(csv_open(StringIO("1,2,3\n4,5,6\nα,β,γ"), self.test_csv_columns)),
            self.test_csv_response
        )

    def test_different_column_order(self):
        self.assertEqual(
            list(csv_open(self.test_csv_path, ('b', 'a', 'c'))),
            self.test_csv_response
        )

    def test_missing_columns(self):
        with self.assertRaises(IndexError):
            list(csv_open(StringIO("1,2"), self.test_csv_columns))
