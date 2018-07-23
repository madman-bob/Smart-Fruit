from collections import namedtuple

from numpy import isfinite
from pandas import Series

from smart_fruit.feature_types.feature_type_base import FeatureType

__all__ = ["Number", "Integer", "Complex", "Label"]


class Number(FeatureType):
    def validate(self, value):
        value = float(value)

        if not isfinite(value):
            raise ValueError(
                "May not assign non-finite value {} to a {}".format(
                    value,
                    self.__class__.__name__
                )
            )

        return value


class Integer(Number):
    def validate(self, value):
        return int(round(super().validate(value)))

    def from_series(self, features):
        return int(round(super().from_series(features)))


class Complex(FeatureType):
    feature_count = 2

    def validate(self, value):
        value = complex(value)

        if not isfinite(value):
            raise ValueError(
                "May not assign non-finite value {} to a {}".format(
                    value,
                    self.__class__.__name__
                )
            )

        return value

    def to_series(self, value):
        return Series([value.real, value.imag])

    def from_series(self, features):
        return complex(*features)


class Label(FeatureType, namedtuple('Label', ['labels'])):
    @property
    def feature_count(self):
        return len(self.labels)

    def validate(self, value):
        if value not in self.labels:
            raise TypeError(
                "May not use non-existent label {!r} in a {!r}".format(
                    value,
                    self
                )
            )

        return value

    def to_series(self, value):
        return Series([int(value == label) for label in self.labels])

    def from_series(self, features):
        return max(zip(features, enumerate(self.labels)))[1][1]
