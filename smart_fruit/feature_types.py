from abc import ABCMeta
from collections import namedtuple

from numpy import isfinite
from pandas import Series

__all__ = ["FeatureType", "Number", "Label"]


class FeatureType(metaclass=ABCMeta):
    _index = None
    feature_count = 1

    def __get__(self, instance, owner):
        if instance is None:
            return self

        if self._index is None:
            self._index = next(
                i
                for i, name in enumerate(owner._fields)
                if getattr(owner, name) is self
            )

        return instance[self._index]

    def validate(self, value):
        return value

    def to_series(self, value):
        return Series([value])

    def from_series(self, features):
        return features[0]


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
