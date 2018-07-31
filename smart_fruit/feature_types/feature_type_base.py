from abc import ABCMeta

from pandas import Series

__all__ = ["FeatureType"]


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
        return features.iloc[0]
